def isnumeric(val):
    # python's builtin str.isnumeric() and str.isdigit() do not
    # work with floating point numbers.
    try:
        float(val)
        return True
    except ValueError:
        return False

def debug(message):
    if VERBOSE:
        print(message)

def delete_column(data, n):
    for record in data:
        del record[n]

def print_data(data, n_rows=10, n_cols=20):
    for row in data[:n_rows//2]:
        print(row[:n_cols//2], "...", row[-n_cols//2:])
    for row in data[-n_rows//2:]:
        print(row[:n_cols//2], "...", row[-n_cols//2:])



def normalize(val, min_val, max_val):
    # returns normalized val between 0 and 1
    return ( (val - min_val) / (max_val - min_val) )

def normalize2(val, min_val, max_val):
    # returns normalized val between -1 and 1
    return 2 * ( (val - min_val) / (max_val - min_val) ) - 1

def normalize_columns(data, min_max_columns):
    for col, min_max in enumerate(min_max_columns):
        if min_max[0] == min_max[1]:
            for row in data:
                row[col] = 0
        else:
            for row in data:
                row[col] = normalize2(row[col], *min_max)

def normalize_rows(data, min_max_rows):
    for row, min_max in enumerate(min_max_rows):
        for col, val in enumerate(data[row]):
            data[row][col] = normalize2(val, *min_max)

