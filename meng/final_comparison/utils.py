import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('x_i', type=int, nargs='?', default=11)
    parser.add_argument('sensor_class', type=float, nargs='?', default=0.5)
    parser.add_argument('seed', type=int, nargs='?', default=1)

    args = parser.parse_args()

    return args