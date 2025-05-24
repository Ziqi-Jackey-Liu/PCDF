import numpy as np


def process_trip_dict(data, min_size=20):
    grouped = data.groupby('trip_id')[['latitude', 'longitude', 'timestamp']]
    trip_dict = {trip_id: group[['latitude', 'longitude', 'timestamp']].values.tolist() for trip_id, group in grouped}

    new_trip_dict = {}
    new_trip_id = 1

    for trip_id, coords_list in trip_dict.items():
        # if element number smaller than min_size, skip
        if len(coords_list) < min_size:
            continue

        # Process the list grouped by each trip_id
        while len(coords_list) >= min_size:
            # Take the first min_size elements as a group and add them to the results
            new_trip_dict[new_trip_id] = coords_list[:min_size]
            coords_list = coords_list[min_size:]  # Truncate the remaining part
            new_trip_id += 1  # update trip_id

            # If the remaining part is less than min_size, discard it directly
            if len(coords_list) < min_size:
                break

    rows = []
    for coords_list in new_trip_dict.values():
        rows.append(np.array(coords_list).T)

    # covert to NumPy array
    result_array = np.array(rows, dtype=object)
    return result_array


def normalized(vector):
    mean_value = np.mean(vector)
    normalized_vector = vector - mean_value
    range_value = np.max(normalized_vector) - np.min(normalized_vector)

    if range_value == 0:
        range_value = 1

    result = 2 * normalized_vector / range_value
    return result


def test(period, cycles, multiplier):
    # Create a periodic vector
    base_pattern = np.arange(1, period + 1)  # Period starting from 1

    # Generate the complete vector based on the number of periods and the scaling factor
    vector = np.concatenate([base_pattern * (multiplier ** i) for i in range(cycles)])
    return vector
