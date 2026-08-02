//! Pure text cleanup matching `manga_ocr.ocr.post_process`.

pub fn post_process(text: &str) -> String {
    let without_whitespace = text
        .chars()
        .filter(|&character| !is_python_whitespace(character))
        .collect::<String>();
    let replaced_ellipsis = without_whitespace.replace('…', "...");

    let mut collapsed = String::with_capacity(replaced_ellipsis.len());
    let mut characters = replaced_ellipsis.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '・' && character != '.' {
            collapsed.push(character);
            continue;
        }

        let mut run_length = 1;
        while let Some(&next) = characters.peek() {
            if next != '・' && next != '.' {
                break;
            }
            characters.next();
            run_length += 1;
        }

        if run_length >= 2 {
            collapsed.extend(std::iter::repeat_n('.', run_length));
        } else {
            collapsed.push(character);
        }
    }

    collapsed
        .chars()
        .map(|character| {
            if ('!'..='~').contains(&character) {
                char::from_u32(character as u32 + 0xFEE0)
                    .expect("printable ASCII maps to fullwidth")
            } else {
                character
            }
        })
        .collect()
}

fn is_python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{001C}'..='\u{001F}')
}
