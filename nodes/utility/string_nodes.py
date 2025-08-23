class DaxStringSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": "+"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "split_text"
    CATEGORY = "Utilities"

    def split_text(self, text, delimiter):
        parts = text.split(delimiter)
        parts = [part.strip() for part in parts]
        # Join with unique delimiter that won't conflict
        packed = "<<<SPLIT>>>".join(parts)
        return (packed,)


class DaxGetStringByIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "packed_strings": ("STRING", {}),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string_by_index"
    CATEGORY = "Utilities"

    def get_string_by_index(self, packed_strings, index):
        parts = packed_strings.split("<<<SPLIT>>>")
        if index < len(parts):
            return (parts[index],)
        return ("",)

NODE_CLASS_MAPPINGS = {
    "DaxStringSplitter": DaxStringSplitter,
    "DaxGetStringByIndex": DaxGetStringByIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaxStringSplitter": "String Splitter",
    "DaxGetStringByIndex": "Get String By Index",
}