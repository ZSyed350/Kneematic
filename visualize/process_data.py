def process_line(line: str):
    """Chip, S0, S1, S2, S3, S4, S5, pos"""
    data = line.split(',')
    chip = int(data[0])
    s0 = float(data[1])
    s1 = float(data[2])
    s2 = float(data[3])
    s3 = float(data[4])
    s4 = float(data[5])
    s5 = float(data[6])
    pos = float(data[7])
    return chip, s0, s1, s2, s3, s4, s5, pos

