

def load_charset(ch_size):
    ch_size =ch_size.lower()
    if ch_size == 's':
        with open('charset/charset_s.txt', 'r', encoding='utf-8') as char_txt:
            charset = [ch.strip() for ch in char_txt.readlines()]
    elif ch_size == 'm':
        with open('charset/charset_m.txt', 'r', encoding='utf-8') as char_txt:
            charset = [ch.strip() for ch in char_txt.readlines()]
    elif ch_size == 'l':
        with open('charset/charset_l.txt', 'r', encoding='utf-8') as char_txt:
            charset = [ch.strip() for ch in char_txt.readlines()]
    else:
        raise ValueError
    return charset