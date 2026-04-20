import pykakasi
import re

def run_local_furigana_logic(lrc_content):
    """
    Adds Furigana to a raw LRC string using pykakasi.
    Returns the enriched LRC string.
    """
    kks = pykakasi.kakasi()
    lines = lrc_content.strip().split('\n')
    final_output = []

    for line in lines:
        # Regex to capture [timestamp] and everything after it
        match = re.match(r"^(\[[0-9:.]+\]\s*)(.*)", line)
        
        if match:
            timestamp = match.group(1)
            japanese_lyrics = match.group(2)
            
            # Convert Japanese text to a list of tokens with readings
            result = kks.convert(japanese_lyrics)
            
            furigana_line = ""
            for item in result:
                # If the original word contains Kanji (orig != hira)
                if item['orig'] != item['hira']:
                    furigana_line += f"{item['orig']}({item['hira']})"
                else:
                    furigana_line += item['orig']
            
            final_output.append(f"{timestamp}{furigana_line}")
        else:
            # Keep metadata lines as they are
            final_output.append(line)

    return "\n".join(final_output)