import sys

from ListenAttendSpell import ListenAttendSpell


def main():
    las = ListenAttendSpell()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        model_path = sys.argv[2]
        las.test(model_path, False)
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        model_path = sys.argv[2]
        las.train(10, False, model_path=model_path)
    else:
        las.train(10, False)

if __name__ == '__main__':
    main()