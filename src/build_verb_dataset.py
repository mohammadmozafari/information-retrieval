def line_generator(file_path):
    with open(file_path, encoding='utf-8') as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            yield line

def write_in_file(file_path, items):
    with open(file_path, encoding='utf-8', mode='w') as f:
        for item in items:
            f.write('{}\n'.format(item))

def main():
    file_path = 'VerbList.fa'
    bon_mazi, bon_mozare = set(), set()

    for i, line in enumerate(line_generator(file_path)):
        parts = line.split('\t')
        if len(parts[1]) != 0:
            bon_mazi.add(parts[1])
        if len(parts[2]) != 0:
            bon_mozare.add(parts[2])

    write_in_file('bon_mazi.fa', bon_mazi)
    write_in_file('bon_mozare.fa', bon_mozare)

if __name__ == "__main__":
    main()
