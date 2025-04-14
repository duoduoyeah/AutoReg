# Convert string to latex

def convert_to_latex(string: str) -> str:
    # Find % that don't have \ before it and replace with \%
    result = ""
    i = 0
    while i < len(string):
        if string[i] == '%' and (i == 0 or string[i-1] != '\\'):
            result += '\\%'
        else:
            result += string[i]
        i += 1
    return result