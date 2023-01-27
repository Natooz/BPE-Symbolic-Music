""" Evaluation metrics functions

"""

from typing import Tuple, List, Union
import csv
import logging
from pathlib import Path, PurePath
import json
from math import ceil

import numpy as np
from miditok import MIDITokenizer, CPWord, OctupleMono
from miditok.constants import CHORD_MAPS
from miditoolkit import MidiFile, Note
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from scipy.special import erf

from constants import CONSISTENCY_WINDOWS_LEN, TIME_DIVISION


def concat_lists(lists: List[List]) -> List:
    """Concatenate several lists
    :param lists: list of lists
    :return: the concatenated list
    """
    concatenated = []
    for lst in lists:
        concatenated += lst
    return concatenated


class Metric:
    """

    :param name:
    """

    def __init__(self, name: str, one_dim: bool = False):
        self.name = name
        self.results = None
        self.one_dim = one_dim

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, out_dir: Union[str, Path, PurePath], reset_after: bool = True):
        with open(PurePath(out_dir, self.name).with_suffix('.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            if self.one_dim:
                writer.writerow(self.results)
            else:
                for track in self.results:
                    writer.writerow(track)
        if reset_after:
            self.reset()

    def _save_json(self, out_dir: Union[str, Path, PurePath], reset_after: bool = True):
        with open(PurePath(out_dir, self.name).with_suffix('.json'), 'w') as f:
            json.dump(self.results, f)
        if reset_after:
            self.reset()

    def load(self, file_dir: Union[str, Path, PurePath]):
        if len(self.results) > 0:
            self.reset()
        with open(PurePath(file_dir, self.name).with_suffix('.csv')) as csvfile:
            reader = csv.reader(csvfile)
            if self.one_dim:
                for row in reader:
                    self.results = [float(i) for i in row]
            else:
                self.results = []
                for row in reader:
                    self.results.append([float(i) for i in row])

    def _load_json(self, file_dir: Union[str, Path, PurePath]):
        with open(PurePath(file_dir, self.name).with_suffix('.json')) as f:
            self.results = json.load(f)

    def analyze(self, logger: logging.Logger, *args, **kwargs):
        if isinstance(self.results[0], np.ndarray) or isinstance(self.results[0], list):
            results = np.concatenate([np.array(arr) for arr in self.results])
        else:
            results = np.array(self.results)
        # mea, td = np.mean(results), np.std(results)
        logger.debug(f'mean {self.name}: {np.mean(results):.4f} ± {np.std(results):.4f}')

    def reset(self):
        self.results = []


# ******* PITCH / VELOCITY / DURATION DISTRIBUTIONS *******

def pitch_distribution(notes: List[Note], norm: bool = False) -> np.ndarray:
    """ Returns the pitch distribution of a list notes

    :param notes: list of the Note objects
    :param norm: will normalize the distribution
    :return: the occurrence of each pitch (in notes played)
    """
    distr = np.array([note.pitch for note in notes])
    if norm:
        distr = distr / np.sum(distr)
    return distr


def pitch_distribution_midi(midi: MidiFile, norm: bool = False) -> List[np.ndarray]:
    """ Returns the pitch distribution of a MIDI (all tracks considered).

    :param midi: the MIDI object
    :param norm: will normalize the distribution
    :return: the occurrence of each pitches (in notes played)
    """
    return [pitch_distribution(track.notes, norm) for track in midi.instruments]


def velocity_distribution(notes: List[Note], norm: bool = False) -> np.ndarray:
    """ Returns the velocity distribution of a list notes

    :param notes: list of the Note objects
    :param norm: will normalize the distribution
    :return: the occurrence of each velocity (in notes played)
    """
    distr = np.array([note.velocity for note in notes])
    if norm:
        distr = distr / np.sum(distr)
    return distr


def velocity_distribution_midi(midi: MidiFile, norm: bool = False) -> List[np.ndarray]:
    """ Returns the velocity distribution of a MIDI (all tracks considered).

    :param midi: the MIDI object
    :param norm: will normalize the distribution
    :return: the occurrence of each velocities (in notes played)
    """
    return [velocity_distribution(track.notes, norm) for track in midi.instruments]


def duration_distribution(notes: List[Note], norm: bool = False) -> np.ndarray:
    """ Returns the duration distribution of a list notes

    :param notes: list of the Note objects
    :param norm: will normalize the distribution
    :return: the distribution of durations (in notes played)
    """
    distr = np.array([note.get_duration() for note in notes])
    if norm:
        distr = distr / np.sum(distr)
    return distr


def duration_distribution_midi(midi: MidiFile, norm: bool = False) -> List[np.ndarray]:
    """ Returns the duration distribution of a MIDI (all tracks considered).

    :param midi: the MIDI object
    :param norm: will normalize the distribution
    :return: the occurrence of each pitches (in notes played)
    """
    return [duration_distribution(track.notes, norm) for track in midi.instruments]


def pitch_vel_dur_distributions(notes: List[Note], norm: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Returns the pitch, velocity and duration distributions of a list notes

    :param notes: the MIDI object
    :param norm: will normalize the distribution
    :return: the pitch, velocity and duration distributions
    """
    return pitch_distribution(notes, norm), velocity_distribution(notes, norm), duration_distribution(notes, norm)


# ******* PITCH METRICS *******

def pitch_unique_nb(pitch_dist: np.ndarray) -> int:
    """ Returns the number of distinct pitches played, from a pitch distribution

    :param pitch_dist: pitch distribution
    :return: number of distinct pitches played within the distribution
    """
    while pitch_dist.ndim > 1:
        pitch_dist = np.sum(pitch_dist, 0)
    return int(np.sum(pitch_dist > 0))


def pitch_range(notes: List[Note]) -> Tuple[int, int]:
    """ Find the lowest and highest pitch in a track, return them and the pitch range.

    :param notes: list of the notes objects
    :return: pitch range, lowest pitch, highest pitch
    """
    mi = min(note.pitch for note in notes)
    ma = max(note.pitch for note in notes)
    return mi, ma


def pitch_range_midi(midi: MidiFile) -> List[Tuple[int, int]]:
    """ Find the pitch ranges, lowest and highest pitches of the tracks of a MIDI object.

    :param midi: the MIDI object
    :return: pitch ranges, lowest and highest, of every track
    """
    return [pitch_range(track.notes) for track in midi.instruments]


def pitch_classes_histogram(notes: List[Note], normalize: bool = True) -> np.ndarray:
    """Count the occurrences of each pitch class in a list of notes.
    It uses the 12 pitch classes of the chromatic scale.
    It returns two histograms, the first being based only on the distinct played notes.
    The second measure the occurrences of each pitch class based on the durations of each note.

    :param notes: list of the notes objects
    :param normalize: normalizes the histogram such that the sum of bin values is 1
    :return: histogram based on distinct played notes, histogram based on the durations of all notes
    """
    pitch_dist = pitch_distribution(notes)  # 1-dim array
    histogram_notes = np.zeros(12)
    for pitch in pitch_dist:
        histogram_notes[pitch.astype(int) % 12] += 1
    if normalize:
        histogram_notes /= sum(histogram_notes)
    return histogram_notes


def pitch_class_histogram_midi(midi: MidiFile, normalize: bool = True) -> list:
    """ Calculate pitch class occurrences of each tracks of a MIDI object.
    For each track, it returns two histograms:
    The first being based only on the distinct played notes.
    The second measure the the occurrences of each pitch class based on the durations of each notes.

    :param midi: the MIDI object
    :param normalize: normalizes the histogram such that the sum of bin values is 1
    :return: histograms of the pitch class of the tracks
    """
    return [pitch_classes_histogram(track.notes, normalize) for track in midi.instruments]


def pitch_class_transition_matrix(notes: List[Note], time_division: int, normalize: bool = False) -> np.ndarray:
    """ Credits to Colin Raffel, code taken and adapted from the Pretty Midi python package:
    https://github.com/craffel/pretty-midi
    Calculate the pitch class transition matrix of a track.
    It contains useful information for tasks such as key detection,
    chord recognition, or genre pattern recognition.
    The two-dimensional pitch class transition matrix is a histogram-like representation computed
    by counting the pitch transitions for each (ordered) pair of notes.

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :param normalize: normalizes the histogram such that the sum of bin values is 1.
    :return: the pitch class transition matrix, shape of (12, 12)
    """
    # less than one note, return all zeros
    if len(notes) <= 1:
        return np.zeros((12, 12))

    # retrieve note starts, ends and pitch classes(nodes)
    starts, ends, nodes = np.array([[x.start, x.end, x.pitch % 12] for x in notes]).T

    # compute distance matrix for all start and end time pairs
    dist_mat = np.subtract.outer(ends, starts)

    # find indices of pairs of notes where the end time of one note is
    # within time_thresh of the start time of the other
    time_thresh = time_division / 8
    sources, targets = np.where(abs(dist_mat) < time_thresh)

    transition_matrix, _, _ = np.histogram2d(nodes[sources], nodes[targets], bins=np.arange(13), normed=normalize)
    return transition_matrix


def pitch_class_transition_matrix_midi(midi: MidiFile, normalize: bool = False) -> List[np.ndarray]:
    """ Calculate the pitch class transition matrix for each track of a MIDI.

    :param midi: the MIDI object
    :param normalize: normalizes the histogram such that the sum of bin values is 1.
    :return: the pitch class transition matrix, shape of (12, 12)
    """
    return [pitch_class_transition_matrix(track, midi.ticks_per_beat, normalize) for track in midi.instruments]


def pitch_shifts(notes: List[Note], time_division: int) -> np.ndarray:
    """ Calculate the intervals (in semitone) between two consecutive played notes in a track.

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :return: average value of consecutive pitch intervals
    """
    if len(notes) <= 1:
        return np.empty(0)
    sources, targets = get_consecutive_notes(notes, time_division)
    intervals = []
    for i in range(len(sources)):
        src_pitch = notes[sources[i]].pitch
        tgt_pitch = notes[targets[i]].pitch
        intervals.append(tgt_pitch - src_pitch)
    if len(intervals) == 0:
        return np.empty(0)
    return np.array(intervals)


def pitch_shifts_midi(midi: MidiFile) -> Union[List[np.ndarray], List[None]]:
    """ Calculate the intervals (in semitone) between two consecutive played notes of a MIDI object.
    For each track, it returns two types of averages:
    The first in the real average, of positive and negative shifts;
    The second is the "absolute" average between each notes.

    :param midi: the MIDI object
    :return: pitch shift averages of each track
    """
    return [pitch_shifts(track.notes, midi.ticks_per_beat) for track in midi.instruments]


def get_consecutive_notes(notes: List[Note], time_division: int) -> Tuple[List[int], List[int]]:
    """ Detect the consecutive notes being played.
    Returns two lists. The first is the notes, the second is the following notes of the first list.
    This way, the note target[i] will be played after the note source[i].

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :return: source notes, target notes
    """
    if len(notes) <= 1:  # less than one note, return zeros
        return [], []
    starts, ends = np.array([[x.start, x.end] for x in notes]).T
    dist_mat = np.subtract.outer(ends, starts)
    # find indices of pairs of notes where the end time of one note is within time_thresh of the start time of the other
    time_thresh = time_division / 8
    sources, targets = np.where(abs(dist_mat) <= time_thresh)
    return sources.tolist(), targets.tolist()


def polyphonic_ratio(notes: List[Note], time_division: int, beat_res: int = 4, onset_offset: int = 1) -> float:
    """ Calculate the polyphonic ratio, i.e. the ratio of notes starting concurrently with other notes.
    NOTE: make sure to sort notes by start time then pitch before: notes.sort(key=lambda x: (x.start, x.pitch))

    :param notes: notes to analyse (sorted by starting time, them pitch)
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param beat_res: beat resolution, i.e. nb of samples per beat (default 4)
    :param onset_offset: maximum offset (in samples / frames / positions) ∈ N separating notes starts to consider them
                            starting at the same time / onset (default 1)
    :return: the polyphonic ratio, calculated as polyphonic_onsets / nb_of_onsets
    """
    poly = n = onset_count = 0
    simul_notes_limit = 20
    onset_offset = time_division * onset_offset / beat_res

    tuples = []
    for note in notes:
        tuples.append((note.pitch, int(note.start), int(note.end)))
    notes = np.asarray(tuples)

    while n < len(notes):
        # Gathers the notes around the same time step
        onset_notes = notes[n:n + simul_notes_limit]  # reduces the scope
        onset_notes = onset_notes[np.where(onset_notes[:, 1] <= onset_notes[0, 1] + onset_offset)]

        if len(onset_notes) > 1:
            poly += 1
        n += len(onset_notes)
        onset_count += 1

    return poly / onset_count


# ******* RHYTHM METRICS *******

def notes_density(notes: List[Note], time_division: int, per_position: bool = False) -> float:
    """ Calculates the note density of a track in notes/beat

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :param per_position: if true, the notes being played at the same time are counted for one
    :return: the note density in notes/beat. Multiply it by the nb of beats per bar (time signature) to get notes/beat.
    """
    if len(notes) == 0:
        return 0
    if per_position:
        return len(set([note.start for note in notes])) / (max(note.end for note in notes) / time_division)
    return len(notes) / (max(note.end for note in notes) / time_division)


def onset_distribution(notes: List[Note], time_division: int) -> np.ndarray:
    """Returns the onsets of a list of notes, in ticks and over 4 beats (4/* time signatures).

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :return: onset distribution
    """
    onsets = []
    ticks_per_bar = 4 * time_division
    for note in notes:
        onsets.append(note.start % ticks_per_bar)
    return np.array(onsets)


def onset_interval(notes: List[Note], time_division: int) -> np.ndarray:
    """ Calculate the onset intervals (intervals between the
    start of two consecutive notes) of a track.
    NOTE: only intervals > 0 and <= 4 beats are considered

    :param notes: list of the notes objects
    :param time_division: MIDI time division / resolution, in ticks/beat
    :return: onset intervals in ticks
    """
    ticks_per_bar = time_division * 4
    intervals = []
    for i in range(len(notes) - 1):
        onset_inter = int(notes[i + 1].start - notes[i].start)
        if 0 < onset_inter <= ticks_per_bar:  # notes played together or too far apart are discarded
            intervals.append(onset_inter)
    return np.array(intervals)


def onset_interval_midi(midi: MidiFile) -> List[np.ndarray]:
    """ Calculate the onset intervals (intervals between the
    start of two consecutive notes) of each track of a MIDI object.

    :param midi: the MIDI object
    :return: onset intervals in ticks
    """
    return [onset_interval(track.notes, midi.ticks_per_beat) for track in midi.instruments]


# ******* ACCURACY / F1 *******

def f1_score(predicted: List[Note], expected: List[Note]) -> float:
    """F1 score between predicted and expected notes.
    The score is based on the duration of the notes
    Here these values corresponds to the sum of durations of their respective occurrences.
    Make sure both predicted and expected notes are ordered by their .start value, then pitch

    :param predicted: the predicted notes
    :param expected: the expected notes
    :return: the f1 score, as TP / (TP + (FP + FN) / 2)
    """
    true_positives = false_positives = false_negatives = 0

    e_idx = 0  # idx of expected notes
    for note_p in predicted:  # true positives and false positives
        matched = False
        for e, note_e in enumerate(expected[e_idx:]):
            if note_e.pitch == note_p.pitch and notes_played_together(note_p, note_e):
                matching_time = min(note_p.end, note_e.end) - max(note_p.start, note_e.start)
                if matching_time > (note_p.end - note_p.end) / 2:
                    matched = True  # considered a match
                    false_negatives += e - e_idx  # passed notes with no match
                    e_idx += e
                    break
            elif note_e.start > note_p.end:
                e_idx += e
                break  # exceeding time
        if matched:
            true_positives += 1
        else:
            false_positives += 1
    return true_positives / (true_positives + (false_positives + false_negatives) / 2)


def notes_played_together(note1: Note, note2: Note) -> bool:
    return note1.end > note2.start and note1.start < note2.end


# ******* ADDITIONAL TOKENS *******

def chord_accuracy(tokens: List[int], tokenizer: MIDITokenizer, onset_offset: int = 1) -> Tuple[float, float]:
    """ Will search for chord tokens in a sequence of generated tokens, and analyze
    if the following tokens represents a chord corresponding to the token's value.
    This only works for MIDI-Like and REMI token sequences.

    :param tokens: the predicted notes
    :param tokenizer: the associated tokenizer
    :param onset_offset: maximum offset (in samples) ∈ N separating notes starts to consider them
                            starting at the same time / onset (default is 1).
                            You should use the same value as the one used to detect detect chords with MidiTok.
    :return: the chord relevance
    """
    err_pitch = 0
    err_len = 0
    chord_idx = [i for i in range(len(tokens)) if tokenizer.vocab.token_type(tokens[i]) == 'Chord']
    if len(chord_idx) == 0:
        return 0, 0

    if tokenizer.__class__.__name__ in ['REMI', 'BarPosDuration']:
        for idx in chord_idx:
            chord = tokenizer.vocab[tokens[idx]].split('_')[1]
            chord_map = CHORD_MAPS[chord] if chord not in ['3', '4', '5'] else [None] * int(chord)
            chord_pos = int(tokenizer.vocab[tokens[idx - 1]].split('_')[1])
            current_pos = chord_pos
            generated_chord = []
            durations = []
            i = 1
            while current_pos - chord_pos <= onset_offset and i + idx < len(tokens):
                token_type, token_value = tokenizer.vocab[tokens[idx + i]].split('_')
                if token_type == 'Pitch':
                    generated_chord.append(int(token_value))
                    beat, pos, res = map(int, tokenizer.vocab[tokens[idx + i + 2]].split('_')[1].split('.'))
                    durations.append(beat * res + pos)
                elif token_type == 'Bar':
                    current_pos = max(tokenizer.beat_res.values()) * 4
                elif token_type == 'Position':
                    if len(durations) > 0 and durations[0] <= max(tokenizer.beat_res.values()) // 2:
                        break
                    current_pos = (current_pos //
                                   (max(tokenizer.beat_res.values()) * 4)) * max(tokenizer.beat_res.
                                                                                 values()) * 4 + int(token_value)
                i += 1

            generated_chord.sort()
            generated_chord = tuple(pitch - generated_chord[0] for pitch in generated_chord)
            if len(generated_chord) != len(chord_map):
                err_len += 1
            elif chord_map[0] is not None and generated_chord != chord_map:
                err_pitch += 1

    elif tokenizer.__class__.__name__ in ['MIDILike', 'TimeShiftBeat']:
        for idx in chord_idx:
            chord = tokenizer.vocab[tokens[idx]].split('_')[1]
            chord_map = CHORD_MAPS[chord] if chord not in ['3', '4', '5'] else [None] * int(chord)
            chord_pos = current_pos = 0
            generated_chord = []
            i = 1
            while current_pos - chord_pos <= onset_offset and i + idx < len(tokens):
                token_type, token_value = tokenizer.vocab[tokens[idx + i]].split('_')
                if token_type == 'Note-On':
                    generated_chord.append(int(token_value))
                elif token_type == 'Time-Shift':
                    beat, pos, res = map(int, token_value.split('.'))
                    current_pos += beat * res + pos
                if token_type == 'Note-Off' and int(token_value) in generated_chord and \
                        current_pos - chord_pos <= max(tokenizer.beat_res.values()) // 2:
                    break  # short chords, as defined in MidiTok
                i += 1

            generated_chord.sort()
            generated_chord = tuple(pitch - generated_chord[0] for pitch in generated_chord)
            if len(generated_chord) != len(chord_map):
                # tokens = [tokenizer.vocab[t].split('_') for t in tokens[idx-1: idx+20]]  # for debug
                err_len += 1
            elif chord_map[0] is not None and generated_chord != chord_map:
                err_pitch += 1
    else:
        raise ValueError('Bad tokenizer, this method only works for MIDI-Like and REMI')

    return err_pitch / len(chord_idx), err_len / len(chord_idx)


def rest_accuracy(tokens: List[int], tokenizer: MIDITokenizer) -> float:
    """ Will search for rests tokens in a sequence of generated tokens, and analyze
    if the detected rests are associated with a token (the model did not produced rests
    without rest tokens)

    :param tokens: the predicted notes
    :param tokenizer: the associated tokenizer
    :return: the chord relevance
    """
    generated_rests = []  # rest tokens
    real_rests = []  # rests spotted when converting tokens to notes
    time_division = max(tokenizer.beat_res.values())
    min_rest = time_division // tokenizer.additional_tokens['rest_range'][0]
    # max_rest = time_division * (tokenizer.additional_tokens['rest_range'][1] + 1) - min_rest
    notes = tokenizer.tokens_to_track(tokens, time_division=time_division)[0].notes

    # 1. Looks for rest tokens (generated_rests)
    if tokenizer.__class__.__name__ in ['REMI', 'BarPosDuration']:
        ticks_per_bar = time_division * 4
        current_tick = 0
        current_bar = -1
        previous_note_end = 0
        for t, token in enumerate(tokens):
            token_type, token_val = tokenizer.vocab.token_to_event[token].split('_')
            if token_type == 'Bar':
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif token_type == 'Rest':
                beat, pos = map(int, token_val.split('.'))
                if current_tick < previous_note_end:  # if in case successive rest happen
                    current_tick = previous_note_end
                generated_rests.append((current_tick, current_tick + beat * time_division + pos))
                current_tick += beat * time_division + pos
                current_bar = current_tick // ticks_per_bar
            elif token_type == 'Position':
                current_tick = current_bar * ticks_per_bar + int(token_val)
            elif token_type == 'Pitch':
                try:
                    token_1_type, _ = tokenizer.vocab.token_to_event[tokens[t + 1]].split('_')
                    token_2_type, token_2_val = tokenizer.vocab.token_to_event[tokens[t + 2]].split('_')
                    if token_1_type == 'Velocity' and token_2_type == 'Duration':
                        beat, pos, res = map(int, token_2_val.split('.'))
                        duration = (beat * res + pos) * time_division // res
                        previous_note_end = max(previous_note_end, current_tick + duration)
                except IndexError as _:  # A well constituted sequence should not raise an exception
                    pass  # However with generated sequences this can happen, or if the sequence isn't finished

    elif tokenizer.__class__.__name__ in ['MIDILike', 'TimeShiftBeat']:
        current_tick = 0
        for token in tokens:
            token_type, token_val = tokenizer.vocab.token_to_event[token].split('_')
            if token_type == 'Time-Shift':
                beat, pos, res = map(int, token_val.split('.'))
                current_tick += (beat * res + pos) * time_division // res
            elif token_type == 'Rest':
                beat, pos = map(int, token_val.split('.'))
                generated_rests.append((current_tick, current_tick + beat * time_division + pos))
                current_tick += beat * time_division + pos

    else:
        raise ValueError('Bad tokenizer, this method only works for MIDI-Like and REMI')

    # 2. Ties successive rests together
    generated_rests_tied = [generated_rests[0]]
    for rest in generated_rests[1:]:
        if rest[0] == generated_rests_tied[-1][1]:
            generated_rests_tied[-1] = (generated_rests_tied[-1][0], rest[1])
        else:
            generated_rests_tied.append(rest)

    # 3. Looks for rests within the generated notes (real_rests)
    previous_note_end = notes[0].end
    for note in notes[1:]:
        if note.start - previous_note_end >= min_rest:
            real_rests.append((previous_note_end, note.start))
        previous_note_end = max(previous_note_end, note.end)

    return 1 - len(generated_rests_tied) / len(real_rests)


# ******* MATHS *******

def mean_std(arr, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the mean and standard deviation of a distribution

    :param arr: distribution, can be a list, tuple or ndarray
    :param axis: axis considered
    :return: mean and standard deviation of the distribution
    """
    return np.mean(arr, axis=axis), np.std(arr, axis=axis)


def histogram_borders(hists: list) -> Tuple[int, int]:
    """ Finds the lowest and highest > 0 index in two histograms

    :param hists: histograms, is a list of either lists of ints or np.ndarrays of dim=1
    :return: lowest index, highest index
    """
    return min(np.min(np.where(hist > 0)) for hist in hists), max(np.max(np.where(hist > 0)) for hist in hists)


def overlapping_area_kde(dist_1: np.ndarray, dist_2: np.ndarray) -> float:
    """ Implementation from: https://github.com/RichardYang40148/mgeval/blob/master/mgeval/core.py
    Computes the overlapping area of the PDF of two distributions, uni or multi-modal.
    The limits of integral are the minimum and maximum indexes of the two distributions.

    :param dist_1: first distribution
    :param dist_2: second distribution
    :return: the overlapping area
    """
    pdf_1 = gaussian_kde(dist_1)
    pdf_2 = gaussian_kde(dist_2)
    return quad(lambda x: min(pdf_1(x), pdf_2(x)),
                np.min((np.min(dist_1), np.min(dist_2))), np.max((np.max(dist_1), np.max(dist_2))))[0]


def overlapping_area_erf(dist_1: np.ndarray, dist_2: np.ndarray) -> float:
    """Implementation from: https://github.com/magenta/symbolic-music-diffusion/blob/main/utils/metrics.py
    Computes overlapping area of two Gaussian distributions.

    :param dist_1: first distribution
    :param dist_2: second distribution
    :return: the overlapping area
    """
    mu1, var1 = mean_std(dist_1)
    mu2, var2 = mean_std(dist_2)

    idx = mu2 < mu1
    mu_a = mu2 * idx + np.logical_not(idx) * mu1
    mu_b = mu1 * idx + np.logical_not(idx) * mu2
    var_a = var2 * idx + np.logical_not(idx) * var1
    var_b = var1 * idx + np.logical_not(idx) * var2

    c_sqrt_factor = (mu_a - mu_b) ** 2 + 2 * (var_a - var_b) * np.log(
        np.sqrt(var_a + 1e-6) / np.sqrt(var_b + 1e-6))
    c_sqrt_factor = np.sqrt(c_sqrt_factor)
    c = mu_b * var_a - np.sqrt(var_b) * (mu_a * np.sqrt(var_b) +
                                         np.sqrt(var_a) * c_sqrt_factor)
    c = c / (var_a - var_b + 1e-6)

    sqrt_2 = np.sqrt(2)
    oa = 1 - 0.5 * erf((c - mu_a) / (sqrt_2 * np.sqrt(var_a + 1e-6)))
    oa = oa + 0.5 * erf((c - mu_b) / (sqrt_2 * np.sqrt(var_b + 1e-6)))
    return oa


def overlapping_area_hist(dist_1: np.ndarray, dist_2: np.ndarray) -> float:
    """Calculates the overlapping area of the histograms of two distributions of integers.

    :param dist_1: first integer distribution
    :param dist_2: second integer distribution
    :return:
    """
    bins = range(min(np.amin(dist_1), np.amin(dist_2)), max(np.amax(dist_1), np.amax(dist_2)) + 1)
    hist1 = np.histogram(dist_1, bins=bins, density=True)[0]
    hist2 = np.histogram(dist_2, bins=bins, density=True)[0]
    return sum([min(hist1[i], hist2[i]) for i in range(len(hist1))])


def note_density_evolution(notes: List[Note], time_division: int, first_beat: int = 0) \
        -> List[float]:
    """
    NOTE: make sure to sort notes by start time then pitch before: notes.sort(key=lambda x: (x.start, x.pitch))

    :param notes: the notes to analyze
    :param time_division: time division of the MIDI
    :param first_beat: first beat to analyze from the notes
    :return: pitch similarities and duration similarities, as lists of scores in [0, 1]
    """
    beat_windows = [[] for _ in range(ceil(max(note.end for note in notes) / time_division) - first_beat)]

    # groups notes per beats
    for note in notes:
        if note.start // time_division < first_beat:
            continue
        beat_windows[int(note.start // time_division)].append(note)

    # groups beats per window length
    beat_windows = [concat_lists(beat_windows[i:i + 4]) for i in range(0, len(beat_windows) + len(beat_windows) % 4, 4)]

    # deletes first and last empty beat windows if there is any
    while len(beat_windows[0]) == 0:
        del beat_windows[0]
    while len(beat_windows[-1]) == 0:
        del beat_windows[-1]

    last_note_density = notes_density(beat_windows[0], time_division, per_position=False)
    note_densities_evolutions = []
    for i in range(1, len(beat_windows)):
        note_density = 0 if len(beat_windows[i]) == 0 \
            else notes_density(beat_windows[i], time_division, per_position=False)
        note_densities_evolutions.append(note_density - last_note_density)
        last_note_density = note_density

    return note_densities_evolutions


def consistency(notes: List[Note], time_division: int, windows_len: int, first_beat: int = 0) \
        -> Tuple[List[float], List[float], List[float]]:
    """
    NOTE: make sure to sort notes by start time then pitch before: notes.sort(key=lambda x: (x.start, x.pitch))

    :param notes: the notes to analyze
    :param time_division: time division of the MIDI
    :param windows_len: size in beats of windows
    :param first_beat: first beat to analyze from the notes
    :return: pitch similarities and duration similarities, as lists of scores in [0, 1]
    """
    beat_windows = [[] for _ in range(ceil(max(note.end for note in notes) / time_division) - first_beat)]

    # groups notes per beats
    for note in notes:
        if note.start // time_division < first_beat:
            continue
        beat_windows[int(note.start // time_division)].append(note)

    # groups beats per window length
    if windows_len > 1:
        beat_windows = [concat_lists(beat_windows[i:i + windows_len])
                        for i in range(0, len(beat_windows) + len(beat_windows) % windows_len, windows_len)]

    # deletes first and last empty beat windows if there is any
    while len(beat_windows[0]) == 0:
        del beat_windows[0]
    while len(beat_windows[-1]) == 0:
        del beat_windows[-1]

    # compute similarities between consecutive windows of beats
    # (overlapping areas of PDF of gaussian distributions of pitches in windows)
    pitch_similarities = []
    velocity_similarities = []
    duration_similarities = []
    for i in range(len(beat_windows) - 1):
        if len(beat_windows[i]) == 0 or len(beat_windows[i + 1]) == 0:
            pitch_similarities.append(0.)
            duration_similarities.append(0.)
            duration_similarities.append(0.)
            continue
        pitch_sim = overlapping_area_hist(pitch_distribution(beat_windows[i]),
                                          pitch_distribution(beat_windows[i + 1]))
        dur_sim = overlapping_area_hist(duration_distribution(beat_windows[i]),
                                        duration_distribution(beat_windows[i + 1]))
        vel_sim = overlapping_area_hist(velocity_distribution(beat_windows[i]),
                                        velocity_distribution(beat_windows[i + 1]))
        pitch_similarities.append(pitch_sim)
        velocity_similarities.append(vel_sim)
        duration_similarities.append(dur_sim)
    return pitch_similarities, velocity_similarities, duration_similarities


class Consistency(Metric):
    def __init__(self, name: str = 'consistency', window_len: int = CONSISTENCY_WINDOWS_LEN):
        super().__init__(name)
        self.window_len = window_len
        self.pitch_results = []  # List[List[float]]
        self.duration_results = []  # List[List[float]]
        self.onset_results = []  # List[List[float]]

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION):
        pitch, duration, onset = consistency(notes, time_division, self.window_len)
        self.pitch_results.append(pitch)
        self.duration_results.append(duration)
        self.onset_results.append(onset)
        return pitch, duration, onset

    def save(self, out_dir: Union[str, Path, PurePath], reset_after: bool = True):
        with open(PurePath(out_dir, 'consistency_pitch.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            for track in self.pitch_results:
                writer.writerow(track)
        with open(PurePath(out_dir, 'consistency_duration.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            for track in self.duration_results:
                writer.writerow(track)
        with open(PurePath(out_dir, 'consistency_onset.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            for track in self.onset_results:
                writer.writerow(track)
        if reset_after:
            self.reset()

    def load(self, file_dir: Union[str, Path, PurePath]):
        if len(self.pitch_results) > 0:
            self.reset()
        with open(PurePath(file_dir, 'consistency_pitch').with_suffix('.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.pitch_results.append([float(i) for i in row])
        with open(PurePath(file_dir, 'consistency_duration').with_suffix('.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.duration_results.append([float(i) for i in row])
        with open(PurePath(file_dir, 'consistency_onset').with_suffix('.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.onset_results.append([float(i) for i in row])

    def analyze(self, logger: logging.Logger, **kwargs):
        pitch_results = np.concatenate([np.array(arr) for arr in self.pitch_results])
        duration_results = np.concatenate([np.array(arr) for arr in self.duration_results])
        onset_results = np.concatenate([np.array(arr) for arr in self.onset_results])
        logger.debug(f'mean pitch consistence: {np.mean(pitch_results):.4f} ± {np.std(pitch_results):.4f}')
        logger.debug(f'mean duration consistence: {np.mean(duration_results):.4f} ± {np.std(duration_results):.4f}')
        logger.debug(f'mean onset consistence: {np.mean(onset_results):.4f} ± {np.std(onset_results):.4f}')

    def reset(self):
        self.pitch_results = []
        self.duration_results = []
        self.onset_results = []


class PitchClass(Metric):
    def __init__(self, name: str = 'pitch_class_histogram'):
        super().__init__(name)
        self.results = []

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> np.ndarray:
        pitch_class_hist = pitch_classes_histogram(notes)
        self.results.append(pitch_class_hist)
        return pitch_class_hist

    def analyze(self, logger: logging.Logger, *args, **kwargs):
        pass


class PitchClassTransition(Metric):
    def __init__(self, name: str = 'pitch_class_transition'):
        super().__init__(name)
        self.results = []  # shape (*, 12, 12)

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> np.ndarray:
        pitch_class_transition = pitch_class_transition_matrix(notes, time_division)
        self.results.append(pitch_class_transition)
        return pitch_class_transition

    def save(self, out_dir: Union[str, Path, PurePath], reset_after: bool = True):
        self._save_json(out_dir, reset_after)

    def load(self, file_dir: Union[str, Path, PurePath]):
        self._load_json(file_dir)

    def analyze(self, logger: logging.Logger, *args, **kwargs):
        pass


class NoteDensity(Metric):
    def __init__(self, name: str = 'note_density'):
        super().__init__(name, one_dim=True)
        self.results = []

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> float:
        note_density = notes_density(notes, time_division)
        self.results.append(note_density)
        return note_density


class NoteDensityEvolution(Metric):
    def __init__(self, name: str = 'note_density_evolution'):
        super().__init__(name, one_dim=True)
        self.results = []

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> List[float]:
        note_density_evol = note_density_evolution(notes, time_division)
        self.results.append(note_density_evol)
        return note_density_evol

    def analyze(self, logger: logging.Logger, *args, **kwargs):
        results_mean = np.array([np.mean(np.array(arr)) for arr in self.results])
        results_std = np.array([np.std(np.array(arr)) for arr in self.results])
        results_var = np.array([np.var(np.array(arr)) for arr in self.results])
        logger.debug(f'mean {self.name}: {np.mean(results_mean):.4f} ± {np.mean(results_std):.4f} '
                     f'var:{np.mean(results_var):.4f}')


class PitchShift(Metric):
    def __init__(self, name: str = 'pitch_shifts'):
        super().__init__(name)
        self.results = []

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> np.ndarray:
        pitch_shift = pitch_shifts(notes, time_division)
        self.results.append(pitch_shift)
        return pitch_shift

    def analyze(self, logger: logging.Logger, *args, **kwargs):
        self.results = [[abs(res) for res in track] for track in self.results]
        super().analyze(logger, *args, **kwargs)


class OnsetInterval(Metric):
    def __init__(self, name: str = 'onset_interval'):
        super().__init__(name)
        self.results = []

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> np.ndarray:
        onset_intervals = onset_interval(notes, time_division)
        self.results.append(onset_intervals / time_division)
        return onset_intervals


class Duration(Metric):
    def __init__(self, name: str = 'duration'):
        super().__init__(name)
        self.results = []  # in beats

    def __call__(self, notes: List[Note], time_division: int = TIME_DIVISION) -> np.ndarray:
        durations = duration_distribution(notes)  # in ticks
        self.results.append(durations / time_division)  # in beats
        return durations


class TokenComprehension(Metric):
    def __init__(self, tokenizer, name: str = 'token_comprehension'):
        super().__init__(name, one_dim=True)
        self.tokenizer = tokenizer
        self.err_type = []
        self.err_time = []
        self.err_note_already_played = []
        self.err_note_not_played = []
        self.err_note_no_noteoff = []

    def __call__(self, x: List) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """

        :param x: input array or list of shape (N,T) or (T), T: time steps; N: batches
        :return: the mean token types err
        """
        if isinstance(self.tokenizer, CPWord):
            err_type, err_time, err_note_already_played, err_note_not_played, err_note_no_noteoff = self.err_cpword(x)
        elif isinstance(self.tokenizer, OctupleMono):
            err_type, err_time, err_note_already_played, err_note_not_played, err_note_no_noteoff = self.err_octuple(x)
        else:
            err_type, err_time, err_note_already_played, err_note_not_played, err_note_no_noteoff = self.err_common(x)

        self.err_type.append(err_type)
        self.err_time.append(err_time)
        self.err_note_already_played.append(err_note_already_played)
        self.err_note_not_played.append(err_note_not_played)
        self.err_note_no_noteoff.append(err_note_no_noteoff)
        return err_type, err_time, err_note_already_played, err_note_not_played, err_note_no_noteoff

    def err_common(self, tokens: List[int]):
        # common method
        err_type = 0
        err_time = 0
        err_note = 0
        original_nb_tokens = len(tokens)
        if self.tokenizer.has_bpe:
            tokens = self.tokenizer.decompose_bpe(tokens)
        previous_type = self.tokenizer.vocab.token_type(tokens[0])
        current_pos = -1
        current_pitches = []
        note_tokens_types = ['Pitch', 'NoteOn', 'PitchVel', 'PitchVelDur']

        if previous_type in note_tokens_types:
            if previous_type in ['Pitch', 'NoteOne']:
                pitch_val = int(self.tokenizer.vocab[tokens[0]].split('_')[1])
            else:  # PitchVel or PitchVelDur
                pitch_val = int(self.tokenizer.vocab[tokens[0]].split('_')[1].split('-')[0])
            current_pitches.append(pitch_val)
        elif previous_type == 'Position':
            current_pos = int(self.tokenizer.vocab[tokens[0]].split('_')[1])

        for token in tokens[1:]:
            token_type, token_value = self.tokenizer.vocab.token_to_event[token].split('_')

            # Good token type
            if token_type in self.tokenizer.tokens_types_graph[previous_type]:
                if token_type == 'Bar':  # reset
                    current_pos = -1
                    current_pitches = []
                elif token_type in ['TimeShift', 'Time-Shift', 'Rest']:
                    current_pitches = []
                elif token_type in note_tokens_types:
                    if token_type in ['Pitch', 'NoteOne']:
                        pitch_val = int(token_value)
                    else:  # PitchVel or PitchVelDur
                        pitch_val = int(token_value.split('-')[0])
                    if pitch_val in current_pitches:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches.append(pitch_val)
                elif token_type == 'Position':
                    if int(token_value) <= current_pos and previous_type != 'Rest':
                        err_time += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = []
            # Bad token type
            else:
                err_type += 1
            previous_type = token_type

        return tuple(map(lambda x: x / original_nb_tokens, (err_type, err_time, err_note, 0., 0.)))

    def err_cpword(self, tokens):
        def cp_token_type(tok: List[int]) -> Tuple[str, str]:
            family = self.tokenizer.vocab[0].token_to_event[tok[0]].split('_')[1]
            if family == 'Note':
                return self.tokenizer.vocab[2].token_to_event[tok[2]].split('_')
            elif family == 'Metric':
                return self.tokenizer.vocab[1].token_to_event[tok[1]].split('_')
            elif family == 'None':
                return 'PAD', 'None'
            raise RuntimeError('No token type found, unknown error')

        err_type = 0
        err_time = 0
        err_note = 0
        previous_type, previous_value = cp_token_type(tokens[0])
        current_pos = -1
        current_pitches = []
        if previous_type == 'Pitch' and previous_value != 'Ignore':
            current_pitches.append(int(previous_value))
        elif previous_type == 'Position' and previous_value != 'Ignore':
            current_pos = int(previous_value)

        for token in tokens[1:]:
            token_type, token_value = cp_token_type(token)
            # Good token type
            if token_type in self.tokenizer.tokens_types_graph[previous_type]:
                if token_type == 'Bar':  # reset
                    current_pos = -1
                    current_pitches = []
                elif token_type == 'Position':
                    if token_value == 'Ignore':
                        err_type += 1
                    elif int(token_value) <= current_pos and previous_type != 'Rest':
                        err_time += 1  # token position value <= to the current position
                    else:
                        current_pos = int(token_value)
                        current_pitches = []
                elif token_type == 'Pitch':
                    if 'Ignore' in [token_value,
                                    self.tokenizer.vocab[3].token_to_event[token[3]].split('_')[1],
                                    self.tokenizer.vocab[4].token_to_event[token[4]].split('_')[1]]:
                        err_type += 1
                    elif int(token_value) in current_pitches:
                        err_note += 1  # pitch already played at current position
                    else:
                        current_pitches.append(int(token_value))
            # Bad token type
            else:
                err_type += 1
            previous_type = token_type

        return tuple(map(lambda x: x / len(tokens), (err_type, err_time, err_note, 0., 0.)))

    def err_octuple(self, tokens):
        err_time = 0
        err_note = 0
        err_type = 0
        current_bar = current_pos = -1
        current_pitches = []

        for token in tokens:
            if all(token[i] == self.tokenizer.vocab[i]['PAD_None'] for i in range(len(token))):
                break
            if any(self.tokenizer.vocab[i][token].split('_')[1] == 'None' for i, token in enumerate(token)):
                err_type += 1
                continue
            bar_value = int(self.tokenizer.vocab[4].token_to_event[token[4]].split('_')[1])
            pos_value = int(self.tokenizer.vocab[3].token_to_event[token[3]].split('_')[1])
            pitch_value = int(self.tokenizer.vocab[0].token_to_event[token[0]].split('_')[1])

            # Bar
            if bar_value < current_bar:
                err_time += 1
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = pos_value
                current_pitches = []
            # Position
            elif pos_value < current_pos:
                err_time += 1
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = []

            # Pitch
            if pitch_value in current_pitches:
                err_note += 1
            else:
                current_pitches.append(pitch_value)

        return tuple(map(lambda x: x / len(tokens), (err_type, err_time, err_note, 0., 0.)))

    def save(self, out_path: Union[str, Path, PurePath], reset_after: bool = True):
        with open(PurePath(out_path), 'w', newline='') as f:
            writer = csv.writer(f)
            for i in range(len(self.err_note_no_noteoff)):
                writer.writerow((self.err_type[i],
                                 self.err_time[i],
                                 self.err_note_already_played[i],
                                 self.err_note_not_played[i],
                                 self.err_note_no_noteoff[i]))
        if reset_after:
            self.reset()

    def load(self, file_path: Union[str, Path, PurePath]):
        if len(self.err_type) > 0:
            self.reset()
        with open(PurePath(file_path)) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.err_type.append(float(row[0]))
                self.err_time.append(float(row[1]))
                self.err_note_already_played.append(float(row[2]))
                self.err_note_not_played.append(float(row[3]))
                self.err_note_no_noteoff.append(float(row[4]))

    def analyze(self, logger: logging.Logger, **kwargs):
        mean_tc_type, std_tc_type = np.mean(self.err_type), np.std(self.err_type)
        mean_tc_time, std_tc_time = np.mean(self.err_time), np.std(self.err_time)
        mean_tc_nap, std_tc_nap = np.mean(self.err_note_already_played), np.std(self.err_note_already_played)
        mean_tc_nnp, std_tc_nnp = np.mean(self.err_note_not_played), np.std(self.err_note_not_played)
        mean_tc_nno, std_tc_nno = np.mean(self.err_note_no_noteoff), np.std(self.err_note_no_noteoff)
        logger.debug(f'TC type: {mean_tc_type:.6f} ± {std_tc_type:.6f} | '
                     f'{mean_tc_type * 100:.1f} ± {std_tc_type * 100:.1f} e-2 | '
                     f'{mean_tc_type * 1000:.1f} ± {std_tc_type * 1000:.1f} e-3')
        logger.debug(f'TC time: {mean_tc_time:.6f} ± {std_tc_time:.6f} | '
                     f'{mean_tc_time * 100:.1f} ± {std_tc_time * 100:.1f} e-2 | '
                     f'{mean_tc_time * 1000:.1f} ± {std_tc_time * 1000:.1f} e-3')
        logger.debug(f'TC nap_: {mean_tc_nap:.6f} ± {std_tc_nap:.6f} | '
                     f'{mean_tc_nap * 100:.1f} ± {std_tc_nap * 100:.1f} e-2 | '
                     f'{mean_tc_nap * 1000:.1f} ± {std_tc_nap * 1000:.1f} e-3')
        logger.debug(f'TC nnp_: {mean_tc_nnp:.6f} ± {std_tc_nnp:.6f} | '
                     f'{mean_tc_nnp * 100:.1f} ± {std_tc_nnp * 100:.1f} e-2 | '
                     f'{mean_tc_nnp * 1000:.1f} ± {std_tc_nnp * 1000:.1f} e-3')
        logger.debug(f'TC nnno: {mean_tc_nno:.6f} ± {std_tc_nno:.6f} | '
                     f'{mean_tc_nno * 100:.1f} ± {std_tc_nno * 100:.1f} e-2 | '
                     f'{mean_tc_nno * 1000:.1f} ± {std_tc_nno * 1000:.1f} e-3')
        logger.debug(f'TC type / nap / time: {mean_tc_type * 1000:.1f} ± {std_tc_type * 1000:.1f} & '
                     f'{mean_tc_nap * 1000:.1f} ± {std_tc_nap * 1000:.1f} & '
                     f'{mean_tc_time * 1000:.1f} ± {std_tc_time * 1000:.1f}')

    def reset(self):
        self.err_type = []
        self.err_time = []
        self.err_note_already_played = []
        self.err_note_not_played = []
        self.err_note_no_noteoff = []
