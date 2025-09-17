def calculate_biomechanical_score(frame_count, running_score, current_value, ideal_value, threshold):
    deviation = abs(current_value - ideal_value)
    current_score = 1 - deviation / threshold * 100

    running_score = (running_score * (frame_count - 1)) / frame_count + current_score / frame_count

    return max(0, min(100, running_score))