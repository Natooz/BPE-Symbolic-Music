""" Similar to MIDI-Like but with Duration tokens instead of Note-Off

"""

from typing import List, Tuple, Dict, Optional, Union, Any, cast
from pathlib import Path

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange
from miditok import MIDITokenizer, TokSequence, Event
from miditok.midi_tokenizer import _in_as_seq
from miditok.constants import MIDI_INSTRUMENTS, TEMPO

from constants import PITCH_RANGE, NB_VELOCITIES, ADDITIONAL_TOKENS, BEAT_RES, TIME_DIVISION, SPECIAL_TOKENS


class TSDPlusPVm(MIDITokenizer):
    r"""Similar to TSD but with Program tokens

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
        additional_tokens["Program"] = True
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens, True, params)

    def tokens_to_track(self, tokens: Union[TokSequence, List, np.ndarray, Any],
                        time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        pass

    def track_to_tokens(self, track: Instrument) -> List[Event]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        (can probably be achieved faster with Mido objects)

        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata['time_division']]
        program = track.program if not track.is_drum else -1
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note On / Velocity / Duration
            events.append(
                Event(type="Program", value=program, time=note.start, desc=note.end)
            )
            events.append(Event(type='PitchVel', time=note.start, value=f'{note.pitch}-{note.velocity}',
                                desc=note.end))
            duration = note.end - note.start
            index = np.argmin(np.abs(dur_bins - duration))
            events.append(Event(type='Duration', time=note.start, value='.'.join(map(str, self.durations[index])),
                                desc=f'{duration} ticks'))

        return events

    def _midi_to_tokens(self, midi: MidiFile, *args, **kwargs) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.

        :param midi: the MIDI objet to convert.
        :return: sequences of tokens.
        """
        # Convert each track to tokens
        dur_bins = self._durations_ticks[self._current_midi_metadata['time_division']]
        all_events = []
        for track in midi.instruments:
            all_events += self.track_to_tokens(track)
        all_events.sort(key=lambda x: x.time)

        current_tick = 0
        for i in range(len(all_events)):
            if all_events[i].time != current_tick:
                time_shift = all_events[i].time - current_tick
                index = np.argmin(np.abs(dur_bins - time_shift))
                all_events.append(Event(type='TimeShift', time=current_tick,
                                        value='.'.join(map(str, self.durations[index])), desc=f'{time_shift} ticks'))
                current_tick = all_events[i].time

        all_events.sort(key=lambda x: (x.time, self._order(x)))
        tok_sequence = TokSequence(events=all_events)
        self.complete_sequence(tok_sequence)
        return tok_sequence

    @_in_as_seq()
    def tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            _=None,
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param _: unused, to match parent method signature
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        tokens = cast(TokSequence, tokens)
        midi = MidiFile(ticks_per_beat=time_division)
        assert (
                time_division % max(self.beat_res.values()) == 0
        ), f"Invalid time division, please give one divisible by {max(self.beat_res.values())}"
        tokens = cast(List[str], tokens.tokens)  # for reducing type errors

        # RESULTS
        instruments: Dict[int, Instrument] = {}
        tempo_changes = [TempoChange(TEMPO, -1)]

        current_tick = 0
        for ti, token in enumerate(tokens):
            if token.split("_")[0] == "TimeShift":
                current_tick += self._token_duration_to_ticks(
                    token.split("_")[1], time_division
                )
            elif token.split("_")[0] == "PitchVel":
                try:
                    if (
                            tokens[ti - 1].split("_")[0] == "Program"
                            and tokens[ti + 1].split("_")[0] == "Duration"
                    ):
                        program = int(tokens[ti - 1].split("_")[1])
                        pitch, vel = map(int, tokens[ti].split("_")[1].split('-'))
                        duration = self._token_duration_to_ticks(
                            tokens[ti + 1].split("_")[1], time_division
                        )
                        if program not in instruments.keys():
                            instruments[program] = Instrument(
                                program=0 if program == -1 else program,
                                is_drum=program == -1,
                                name="Drums"
                                if program == -1
                                else MIDI_INSTRUMENTS[program]["name"],
                            )
                        instruments[program].notes.append(
                            Note(vel, pitch, current_tick, current_tick + duration)
                        )
                except (
                        IndexError
                ):  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        # create MidiFile
        midi.instruments = list(instruments.values())
        midi.tempo_changes = tempo_changes
        midi.max_tick = max(
            [
                max([note.end for note in track.notes]) if len(track.notes) > 0 else 0
                for track in midi.instruments
            ]
        )
        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def _create_base_vocabulary(self, sos_eos_tokens: bool = False) -> List[str]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # NOTE ON
        vocab += [f'PitchVel_{i}-{vel}' for i in self.pitch_range for vel in self.velocities]

        # DURATION
        vocab += [
            f'Duration_{".".join(map(str, duration))}' for duration in self.durations
        ]

        # TIME SHIFTS
        vocab += [
            f'TimeShift_{".".join(map(str, self.durations[i]))}'
            for i in range(len(self.durations))
        ]

        # PROGRAMS
        vocab += [f"Program_{program}" for program in self.additional_tokens["programs"]]

        # CHORD
        if self.additional_tokens["Chord"]:
            vocab += self._create_chords_tokens()

        # REST
        if self.additional_tokens["Rest"]:
            vocab += [f'Rest_{".".join(map(str, rest))}' for rest in self.rests]

        # TEMPO
        if self.additional_tokens["Tempo"]:
            vocab += [f"Tempo_{i}" for i in self.tempos]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic["Program"] = ["PitchVel"]
        dic["PitchVel"] = ["Duration"]
        dic["Duration"] = ["Program", "TimeShift"]
        dic["TimeShift"] = ["Program"]

        if self.additional_tokens["Chord"]:
            dic["Chord"] = ["PitchVel"]
            dic["TimeShift"] += ["Program"]

        if self.additional_tokens["Tempo"]:
            dic["TimeShift"] += ["Tempo"]
            dic["Tempo"] = ["Program", "TimeShift"]

        if self.additional_tokens["Rest"]:
            dic["Rest"] = ["Rest", "Program", "TimeShift"]
            dic["Duration"] += ["Rest"]

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        r"""Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == "Tempo":
            return 1
        elif x.type == "TimeShift" or x.type == "Rest":
            return 1000  # always last
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 6
