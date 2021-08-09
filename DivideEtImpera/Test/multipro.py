import multiprocessing
manager = multiprocessing.Manager()



EXAMPLES = [[1,1,1],[0,0,0],[2,2,2],[3,3,3]]
subprob = (0,1)


def fill(l, examp, subprob):
    for e in examp:
        l.append([e[i] for i in subprob])


def main():

    l = manager.list()
    ln = int(len(EXAMPLES) / 2)
    proc1 = multiprocessing.Process(target= fill, args=[l, EXAMPLES[:ln], subprob])
    proc2 = multiprocessing.Process(target= fill, args=[l, EXAMPLES[ln:len(EXAMPLES)], subprob])
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()
    print(l)


if __name__ == '__main__':
    main()
