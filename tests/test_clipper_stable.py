from shorts_generator.local.clipper import _FaceTracker, REQUIRED_CONSECUTIVE_FRAMES


def test_first_detection_becomes_initial_center():
    tracker = _FaceTracker(1000, 1000)
    assert tracker.update((500, 500)) == (500, 500)


def test_no_detection_at_start_defaults_to_frame_center():
    tracker = _FaceTracker(1000, 800)
    assert tracker.update(None) == (500, 400)


def test_small_movement_is_trusted_and_smoothed_immediately():
    tracker = _FaceTracker(1000, 1000)
    tracker.update((500, 500))
    result = tracker.update((520, 500))  # well within MAX_TRUSTED_JUMP_FRACTION
    assert result != (500, 500)  # moved toward the new detection
    assert result[0] < 520  # smoothed, not snapped all the way there


def test_missing_detection_holds_position_steady():
    tracker = _FaceTracker(1000, 1000)
    tracker.update((500, 500))
    held = tracker.update(None)
    assert held == tracker.last_center


def test_single_frame_false_positive_is_ignored():
    tracker = _FaceTracker(1000, 1000)
    tracker.update((500, 500))
    before = tracker.last_center

    # a single wildly-distant detection (e.g. a background object)
    after_outlier = tracker.update((50, 900))
    assert after_outlier == before  # held steady, not snapped to the outlier

    # the real subject reappears right where it was — tracker recovers cleanly
    back = tracker.update((505, 500))
    assert back == (500, 500) or abs(back[0] - 500) <= 1


def test_sustained_jump_is_eventually_trusted():
    tracker = _FaceTracker(1000, 1000)
    tracker.update((500, 500))

    # the same far-away point detected for REQUIRED_CONSECUTIVE_FRAMES in a row
    # (e.g. the subject actually walked across the frame) should be followed
    for _ in range(REQUIRED_CONSECUTIVE_FRAMES - 1):
        result = tracker.update((900, 900))
        assert (
            result == tracker.last_center
        )  # still pending, not yet moved past smoothing step
    final = tracker.update((900, 900))
    assert (
        final[0] > 500 and final[1] > 500
    )  # tracker followed the sustained new target


def test_alternating_outliers_never_accumulate_into_a_switch():
    tracker = _FaceTracker(1000, 1000)
    tracker.update((500, 500))
    before = tracker.last_center

    # different outlier location each time -> pending never reaches
    # REQUIRED_CONSECUTIVE_FRAMES, tracker never switches
    tracker.update((10, 10))
    tracker.update((990, 10))
    tracker.update((10, 990))
    assert tracker.last_center == before
