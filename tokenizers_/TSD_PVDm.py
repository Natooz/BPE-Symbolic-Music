"""Similar to MIDI-Like but with Duration tokens instead of Note-Off

"""

from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditoolkit import Instrument, Note, TempoChange
from miditok import MIDITokenizer, Vocabulary, Event
from miditok.constants import MIDI_INSTRUMENTS, TEMPO

from constants import PITCH_RANGE, NB_VELOCITIES, ADDITIONAL_TOKENS, BEAT_RES, TIME_DIVISION


class TSDPVDm(MIDITokenizer):
    r"""Similar to MIDI-Like but with Duration tokens instead of Note-Off

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (chords, time signature, rests, tempo...)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 pad: bool = True, sos_eos: bool = False, mask: bool = False, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, pad, sos_eos, mask, False, params)

    def track_to_tokens(self, track: Instrument) -> List[int]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self.durations_ticks[self.current_midi_metadata['time_division']]
        events = []

        # Creates the Note On, Note Off and Velocity events
        current_tick = 0
        for n, note in enumerate(track.notes):
            if note.start != current_tick:
                time_shift = note.start - current_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                events.append(Event(type_='Time-Shift', time=current_tick,
                                    value='.'.join(map(str, self.durations[index])), desc=f'{time_shift} ticks'))
                current_tick = note.start
            # Note On / Velocity / Duration
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(Event(type_='PitchVelDur', time=note.start,
                                value=f"{note.pitch}-{note.velocity}-{'.'.join(map(str, self.durations[index]))}",
                                desc=note.end))

        return self.events_to_tokens(events)

    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        events = self.tokens_to_events(tokens)

        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].type == 'PitchVelDur':
                pitch, vel, dur = events[ei].value.split('-')
                pitch, vel = map(int, (pitch, vel))
                duration = self._token_duration_to_ticks(dur, time_division)
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
            elif events[ei].type == 'Time-Shift':
                current_tick += self._token_duration_to_ticks(events[ei].value, time_division)
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self) -> Vocabulary:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :return: the vocabulary object
        """
        vocab = Vocabulary(pad=self._pad, mask=self._mask, sos_eos=self._sos_eos)

        # NOTE ON
        vocab.add_event(f'PitchVelDur_{i}-{vel}-{".".join(map(str, duration))}' for i in self.pitch_range
                        for vel in self.velocities for duration in self.durations)

        # TIME SHIFTS
        vocab.add_event(f'Time-Shift_{".".join(map(str, self.durations[i]))}' for i in range(len(self.durations)))

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['PitchVelDur'] = ['PitchVelDur', 'Time-Shift']
        dic['Time-Shift'] = ['PitchVelDur']

        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(self, tokens: List[int], consider_pad: bool = False) -> Tuple[Union[float, Any]]:
        r"""Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a Pitch token should not be present if the same pitch is already being played

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err_type = 0
        err_note = 0
        previous_type = self.vocab.token_type(tokens[0])
        current_pitches = []
        if previous_type == 'PitchVelDur':
            current_pitches.append(int(self.vocab[tokens[0]].split('_')[1].split('-')[0]))

        def check(tok: int):
            nonlocal err_type, err_note, previous_type, current_pitches
            token_type, token_value = self.vocab.token_to_event[tok].split('_')
            # Good token type
            if token_type in self.tokens_types_graph[previous_type]:
                if token_type == 'PitchVelDur':
                    if int(token_value.split('-')[0]) in current_pitches:
                        err_note += 1  # pitch already being played
                    else:
                        current_pitches.append(int(token_value.split('-')[0]))
                elif token_type == 'Time-Shift':
                    current_pitches = []  # moving in time, list reset
            # Bad token type
            else:
                err_type += 1
            previous_type = token_type

        if consider_pad:
            for token in tokens[1:]:
                check(token)
        else:
            for token in tokens[1:]:
                if previous_type == 'PAD':
                    break
                check(token)

        return tuple(map(lambda x: x / len(tokens), (err_type, 0., err_note, 0., 0.)))
