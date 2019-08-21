import sys

from ListenAttendSpell import ListenAttendSpell


def main():
    las = ListenAttendSpell()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        model_path = sys.argv[2]
        las.test(model_path, True)
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        model_path = sys.argv[2]
        las.train(100, True, model_path=model_path)
    else:
        las.train(100, True)

if __name__ == '__main__':
    main()