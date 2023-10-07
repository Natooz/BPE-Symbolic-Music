""" Similar to MIDI-Like but with Duration tokens instead of Note-Off

"""

from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditoolkit import Instrument, Note, TempoChange
from miditok import MIDITokenizer, TokSequence, Event
from miditok.midi_tokenizer import _out_as_complete_seq, _in_as_seq
from miditok.constants import MIDI_INSTRUMENTS, TEMPO

from constants import PITCH_RANGE, NB_VELOCITIES, ADDITIONAL_TOKENS, BEAT_RES, TIME_DIVISION, SPECIAL_TOKENS


class TSDPVm(MIDITokenizer):
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
                 special_tokens=SPECIAL_TOKENS, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens, False, params)

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata['time_division']]
        events = []

        # Creates the Note On, Note Off and Velocity events
        current_tick = 0
        for n, note in enumerate(track.notes):
            if note.start != current_tick:
                time_shift = note.start - current_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                events.append(Event(type='TimeShift', time=current_tick,
                                    value='.'.join(map(str, self.durations[index])), desc=f'{time_shift} ticks'))
                current_tick = note.start
            # Note On / Velocity / Duration
            events.append(Event(type='PitchVel', time=note.start, value=f'{note.pitch}-{note.velocity}',
                                desc=note.end))
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(Event(type='Duration', time=note.start, value='.'.join(map(str, self.durations[index])),
                                desc=f'{duration} ticks'))

        return TokSequence(events=events)

    @_in_as_seq()
    def tokens_to_track(self, tokens: Union[TokSequence, List, np.ndarray, Any],
                        time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        r"""Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        tokens = tokens.tokens

        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(tokens):
            token_type, token_val = tokens[ei].split("_")
            if token_type == 'PitchVel':
                try:
                    if tokens[ei + 1].split("_")[0] == 'Duration':
                        pitch, vel = map(int, token_val.split('-'))
                        duration = self._token_duration_to_ticks(tokens[ei + 1].split("_")[1], time_division)
                        instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        ei += 1
                except IndexError as _:
                    pass
            elif token_type == 'TimeShift':
                current_tick += self._token_duration_to_ticks(token_val, time_division)
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_base_vocabulary(self) -> List[str]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :return: the vocabulary object
        """
        vocab = []

        # NOTE ON
        vocab += [f'PitchVel_{i}-{vel}' for i in self.pitch_range for vel in self.velocities]

        # DURATION
        vocab += [f'Duration_{".".join(map(str, duration))}' for duration in self.durations]

        # TIME SHIFTS
        vocab += [f'TimeShift_{".".join(map(str, self.durations[i]))}' for i in range(len(self.durations))]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['PitchVel'] = ['Duration']
        dic['Duration'] = ['PitchVel', 'TimeShift']
        dic['TimeShift'] = ['PitchVel']

        return dic
