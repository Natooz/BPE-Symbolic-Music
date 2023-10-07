""" Evaluation metrics functions

"""

from typing import Tuple, List

import numpy as np
from miditok import MIDITokenizer, CPWord, OctupleMono, TokSequence
from miditoolkit import MidiFile, Note
from sklearn.decomposition import PCA


def err_cpword(tokens, tokenizer) -> Tuple[float, float, float, float, float]:
    def cp_token_type(tok: List[int]) -> Tuple[str, str]:
        family = tokenizer[0, tok[0]].split('_')[1]
        if family == 'Note':
            return tokenizer[2, tok[2]].split('_')
        elif family == 'Metric':
            return tokenizer[1, tok[1]].split('_')
        elif family == 'None':
            return 'PAD', 'None'
        raise RuntimeError('No token type found, unknown error')

    err_type = 0
    err_time = 0
    err_note = 0
    if len(tokens) < 2:
        return 0, 0, 0, 0, 0
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
        if token_type in tokenizer.tokens_types_graph[previous_type]:
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
                                tokenizer[3, token[3]].split('_')[1],
                                tokenizer[4, token[4]].split('_')[1]]:
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


def err_octuple(tokens, tokenizer) -> Tuple[float, float, float, float, float]:
    if len(tokens) < 2:
        return 0, 0, 0, 0, 0

    err_time = 0
    err_note = 0
    err_type = 0
    current_bar = current_pos = -1
    current_pitches = []

    for token in tokens:
        if all(token[i] == tokenizer.vocab[i]['PAD_None'] for i in range(len(token))):
            break
        if any(tokenizer[i, tok].split('_')[1] == 'None' for i, tok in enumerate(token)):
            err_type += 1
            continue
        bar_value = int(tokenizer[4, token[4]].split('_')[1])
        pos_value = int(tokenizer[3, token[3]].split('_')[1])
        pitch_value = int(tokenizer[0, token[0]].split('_')[1])

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


def tse(tokens: List[int], tokenizer: MIDITokenizer) -> Tuple[float, float, float, float, float]:
    r"""Checks if a sequence of tokens is made of good token types
    successions and returns the error ratio (lower is better).
    The common implementation in MIDITokenizer class will check token types,
    duplicated notes and time errors. It works for REMI, TSD and Structured.
    Other tokenizations override this method to include other errors
    (like no NoteOff / NoteOn for MIDILike and embedding pooling).
    Overridden methods must call decompose_bpe at the beginning if BPE is used!

    :param tokens: sequence of tokens to check.
    :param tokenizer
    :return: the error ratio (lower is better).
    """
    if isinstance(tokenizer, OctupleMono):
        return err_octuple(tokens, tokenizer)
    elif isinstance(tokenizer, CPWord):
        return err_cpword(tokens, tokenizer)

    nb_tok_predicted = len(tokens)  # used to norm the score
    tokens = TokSequence(ids=tokens, ids_bpe_encoded=tokenizer.has_bpe)
    if tokenizer.has_bpe:
        tokenizer.decode_bpe(tokens)
    tokenizer.complete_sequence(tokens)
    tokens = tokens.tokens

    err_type = 0  # i.e. incompatible next type predicted
    err_time = 0  # i.e. goes back or stay in time (does not go forward)
    err_ndup = 0
    err_nnon = 0  # note-off predicted while not being played
    err_nnof = 0  # note-on predicted with no note-off to end it
    previous_type = tokens[0].split("_")[0]
    current_pos = -1
    notes_being_played = {pitch: 0 for pitch in range(0, 128)}
    pitches_current_moment = []  # only at the current position / time step - used for ndup
    note_tokens_types = ["Pitch", "NoteOn", "PitchVel", "PitchVelDur"]
    pos_per_beat = max(tokenizer.beat_res.values())
    max_duration = tokenizer.durations[-1][0] * pos_per_beat
    max_duration += tokenizer.durations[-1][1] * (pos_per_beat // tokenizer.durations[-1][2])

    # Init first note and current pitches if needed
    if previous_type in note_tokens_types:
        notes_being_played[int(tokens[0].split("_")[1])] += 1
        pitches_current_moment.append(int(tokens[0].split("_")[1]))
    elif previous_type == "Position":
        current_pos = int(tokens[0].split("_")[1])
    del tokens[0]

    for i, token in enumerate(tokens):
        event_type, event_value = token.split("_")

        # Good token type
        if event_type in tokenizer.tokens_types_graph[previous_type]:
            if event_type == "Bar":  # reset
                current_pos = -1
                pitches_current_moment = []

            elif event_type == "Position":
                if int(event_value) <= current_pos and previous_type != "Rest":
                    err_time += 1  # token position value <= to the current position
                current_pos = int(event_value)
                pitches_current_moment = []

            elif event_type == "TimeShift":
                pitches_current_moment = []

            elif event_type in note_tokens_types:  # checks if not already played and/or that a NoteOff is associated
                if event_type in ["Pitch", "NoteOn"]:
                    pitch_val = int(event_value)
                else:  # PitchVel or PitchVelDur
                    pitch_val = int(event_value.split("-")[0])

                if pitch_val in pitches_current_moment:
                    err_ndup += 1  # pitch already played at current position
                pitches_current_moment.append(pitch_val)
                if event_type == "NoteOn":
                    # look for an associated note off token to get duration
                    offset_sample = 0
                    offset_bar = 0
                    for j in range(i + 1, len(tokens)):
                        event_j_type, event_j_value = tokens[j].split("_")[0], tokens[j].split("_")[1]
                        if event_j_type == 'NoteOff' and int(event_j_value) == pitch_val:
                            notes_being_played[pitch_val] += 1
                            break  # all good
                        elif event_j_type == 'Bar':
                            offset_bar += 1
                        elif event_j_type == 'Position':
                            if offset_bar == 0:
                                offset_sample = int(event_j_value) - current_pos
                            else:
                                offset_sample = pos_per_beat - current_pos + (offset_bar - 1) * pos_per_beat * 4 + \
                                                int(event_j_value)
                        elif event_j_type == 'TimeShift':
                            offset_sample += tokenizer._token_duration_to_ticks(event_j_value, pos_per_beat)
                        if offset_sample > max_duration:  # will not look for Note Off beyond
                            err_nnof += 1
                            break

            elif event_type == "NoteOff":
                if notes_being_played[int(event_value)] == 0:
                    err_nnon += 1  # this note wasn't being played
                else:
                    notes_being_played[int(event_value)] -= 1
        # Bad token type
        else:
            err_type += 1
        previous_type = event_type

    return tuple(map(lambda err: err / nb_tok_predicted, (err_type, err_time, err_ndup, err_nnon, err_nnof)))


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


# ******* MATHS *******

def mean_std(arr, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the mean and standard deviation of a distribution

    :param arr: distribution, can be a list, tuple or ndarray
    :param axis: axis considered
    :return: mean and standard deviation of the distribution
    """
    return np.mean(arr, axis=axis), np.std(arr, axis=axis)


def isoscore(points: np.ndarray):
    # adapted from https://github.com/bcbi-edu/p_eickhoff_isoscore

    # Step 2
    pca = PCA(n_components=min(points.shape))
    points_pca = pca.fit_transform(points)

    # Step 3
    n = np.shape(points_pca)[0]
    cov = np.cov(points_pca)
    cov_diag = cov[np.diag_indices(n)]

    # Step 4
    n = len(cov_diag)
    cov_diag_normalized = (cov_diag * np.sqrt(n)) / np.linalg.norm(cov_diag)

    # Step 5
    n = len(cov_diag_normalized)
    iso_diag = np.eye(n)[np.diag_indices(n)]
    l2_norm = np.linalg.norm(cov_diag_normalized - iso_diag)
    normalization_constant = np.sqrt(2 * (n - np.sqrt(n)))
    isotropy_defect = l2_norm / normalization_constant

    # Steps 6 and 7
    n = np.shape(points)[0]
    the_score = ((n - (isotropy_defect ** 2) * (n - np.sqrt(n))) ** 2 - n) / (n * (n - 1))

    return the_score
