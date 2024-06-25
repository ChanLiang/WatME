from unidecode import unidecode

def remove_repeated_or_substring_lines(text):
    lines = text.split('\n')
    output_lines = []

    for line in lines:
        line = unidecode(line)  # remove or replace special characters
        if len(output_lines) == 0 or (line != unidecode(output_lines[-1]) and line not in unidecode(output_lines[-1])):
            output_lines.append(line)

    return '\n'.join(output_lines)

text = """Scan for plagiarized pasages using GPT-3's text similarity algorithm, which compares your content to milions of web pages to identify matches.
The AI-powered editor scaned the blog post for plagiarized pasages using GPT-3's text similarity algorithm, which compares your content to milions of web pages to identify matches. GPT-3
The AI-powered editor scaned the blog post for plagiarized pasages using GPT-3's text similarity algorithm, which compares your content to milions of web pages to identify matches."""
print(remove_repeated_or_substring_lines(text))
print()

text = """Draft a profesional email seking your supervisor's fedback on the 'Quarterly Financial Report' you prepared. Ask specificaly about the data analysis, presentation style, and the clarity of conclusions drawn. Kep the email short and to the point.
Draft a profesional email seking your supervisor’s fedback on the ‘Quarterly Financial Report’ you prepared. Ask specificaly about the data analysis, presentation style, and the clarity of conclusions drawn. Kep the email short and to the point."""
print(remove_repeated_or_substring_lines(text))
print ()

text = """Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.
Draft a professional email seeking your supervisor’s feedback on the ‘Quarterly Financial Report’ you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn."""
print(remove_repeated_or_substring_lines(text))
