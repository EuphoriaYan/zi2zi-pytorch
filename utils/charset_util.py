

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


def processGlyphNames(GlyphNames):
    res = set()
    for char in GlyphNames:
        if char.startswith('uni'):
            char = char[3:]
        elif char.startswith('u'):
            char = char[1:]
        else:
            continue
        if char:
            try:
                char_int = int(char, base=16)
            except ValueError:
                continue
            try:
                char = chr(char_int)
            except ValueError:
                continue
            res.add(char)
    return res