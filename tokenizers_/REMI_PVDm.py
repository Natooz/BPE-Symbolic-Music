""" Like REMI, but with pitch, velocity and duration tokens merged

"""

from typing import List, Tuple, Dict, Union, Any, Optional

from miditok import MIDITokenizer, TokSequence, Event
from miditok.midi_tokenizer import _out_as_complete_seq, _in_as_seq
from miditok.constants import TEMPO, MIDI_INSTRUMENTS
from miditoolkit import Note, Instrument, TempoChange
import numpy as np

from constants import PITCH_RANGE, NB_VELOCITIES, ADDITIONAL_TOKENS, BEAT_RES, TIME_DIVISION, SPECIAL_TOKENS


class REMIPVDm(MIDITokenizer):
    r"""Similar to REMI, but different error type function from the one in MidiTok
    https://arxiv.org/abs/2002.00212

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
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
                 special_tokens=SPECIAL_TOKENS, params=None):
        additional_tokens['Chord'] = False  # Incompatible additional token
        additional_tokens['Rest'] = False
        additional_tokens['Tempo'] = False
        additional_tokens['TimeSignature'] = False
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens, False, params)

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self._current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self._current_midi_metadata['time_division'] * 4
        dur_bins = self._durations_ticks[self._current_midi_metadata['time_division']]
        events = []

        # Creates events
        previous_tick = -1
        current_bar = -1
        for note in track.notes:
            if note.start != previous_tick:
                # Bar
                nb_new_bars = note.start // ticks_per_bar - current_bar
                for i in range(nb_new_bars):
                    events.append(Event(type='Bar', time=(current_bar + i + 1) * ticks_per_bar, value='None', desc=0))
                current_bar += nb_new_bars

                # Position
                pos_index = int((note.start % ticks_per_bar) / ticks_per_sample)
                events.append(Event(type='Position', time=note.start, value=pos_index, desc=note.start))

                previous_tick = note.start

            # Pitch / Velocity / Duration
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            val = f"{note.pitch}-{note.velocity}-{'.'.join(map(str, self.durations[index]))}"
            events.append(Event(type='PitchVelDur', time=note.start, value=val, desc=val))

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
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        tokens = self._ids_to_tokens(tokens.ids)

        ticks_per_sample = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        current_bar = -1
        for ti, token in enumerate(tokens):
            token_type, token_val = token.split("_")
            if token_type == 'Bar':
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif token_type == 'Position':
                if current_bar == -1:
                    current_bar = 0  # as this Position token occurs before any Bar token
                current_tick = current_bar * ticks_per_bar + int(token_val) * ticks_per_sample
            elif token_type == 'PitchVelDur':
                pitch, vel, dur = token_val.split('-')
                pitch, vel = map(int, (pitch, vel))
                duration = self._token_duration_to_ticks(dur, time_division)
                instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))

        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_base_vocabulary(self) -> List[str]:
        r"""Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.

        :return: the vocabulary object
        """
        # BAR
        vocab = ["Bar_None"]

        # PITCH VEL DUR
        vocab += [f'PitchVelDur_{i}-{vel}-{".".join(map(str, duration))}' for i in self.pitch_range
                  for vel in self.velocities for duration in self.durations]

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab += [f'Position_{i}' for i in range(nb_positions)]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['PitchVelDur'] = ['PitchVelDur', 'Bar', 'Position']
        dic['Bar'] = ['Position']
        dic['Position'] = ['PitchVelDur']

        return dic
