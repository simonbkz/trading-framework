//+------------------------------------------------------------------+
//|                                              JsonParser.mqh       |
//|  Lightweight JSON parser for the SignalBridge EA.                  |
//|                                                                   |
//|  Supports: string, number, and array extraction by key name.      |
//|  Does NOT support nested objects — designed for flat signal dicts. |
//+------------------------------------------------------------------+
#property copyright "Trading Framework"
#property strict

//+------------------------------------------------------------------+
//| Extract a string value by key from a JSON string                  |
//| e.g. JsonGetString(json, "asset") -> "EURUSD"                    |
//+------------------------------------------------------------------+
string JsonGetString(string &json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";

   // Move past "key":
   pos += StringLen(search);

   // Skip whitespace and colon
   int len = StringLen(json);
   while(pos < len)
   {
      ushort ch = StringGetCharacter(json, pos);
      if(ch == ':' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
         pos++;
      else
         break;
   }

   if(pos >= len) return "";

   ushort firstChar = StringGetCharacter(json, pos);

   // Quoted string value
   if(firstChar == '"')
   {
      pos++;  // skip opening quote
      string result = "";
      while(pos < len)
      {
         ushort ch = StringGetCharacter(json, pos);
         if(ch == '"') break;
         if(ch == '\\' && pos + 1 < len)
         {
            pos++;
            ch = StringGetCharacter(json, pos);
         }
         result += ShortToString(ch);
         pos++;
      }
      return result;
   }

   // Unquoted value (number, bool, null) — return as string
   string result = "";
   while(pos < len)
   {
      ushort ch = StringGetCharacter(json, pos);
      if(ch == ',' || ch == '}' || ch == ']' || ch == ' ' || ch == '\n' || ch == '\r')
         break;
      result += ShortToString(ch);
      pos++;
   }
   return result;
}

//+------------------------------------------------------------------+
//| Extract a numeric value by key from a JSON string                 |
//| e.g. JsonGetDouble(json, "lots") -> 0.10                         |
//+------------------------------------------------------------------+
double JsonGetDouble(string &json, string key)
{
   string val = JsonGetString(json, key);
   if(val == "" || val == "null") return 0.0;
   return StringToDouble(val);
}

//+------------------------------------------------------------------+
//| Extract a JSON array by key into a string array                   |
//| Each element is a raw JSON string (object or primitive)           |
//|                                                                   |
//| Returns the number of elements found.                             |
//+------------------------------------------------------------------+
int JsonGetArray(string &json, string key, string &items[])
{
   ArrayResize(items, 0);

   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return 0;

   // Find the opening bracket
   pos = StringFind(json, "[", pos);
   if(pos < 0) return 0;
   pos++; // skip '['

   int len = StringLen(json);
   int count = 0;

   while(pos < len)
   {
      // Skip whitespace
      while(pos < len)
      {
         ushort ch = StringGetCharacter(json, pos);
         if(ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == ',')
            pos++;
         else
            break;
      }

      if(pos >= len) break;

      ushort ch = StringGetCharacter(json, pos);

      // End of array
      if(ch == ']') break;

      // Start of object
      if(ch == '{')
      {
         string element = ExtractObject(json, pos);
         if(element != "")
         {
            ArrayResize(items, count + 1);
            items[count] = element;
            count++;
            pos += StringLen(element);
         }
         else
         {
            pos++;
         }
      }
      // Start of string
      else if(ch == '"')
      {
         string element = ExtractQuotedString(json, pos);
         ArrayResize(items, count + 1);
         items[count] = element;
         count++;
         pos += StringLen(element) + 2; // +2 for quotes
      }
      // Primitive value
      else
      {
         string element = "";
         while(pos < len)
         {
            ushort c = StringGetCharacter(json, pos);
            if(c == ',' || c == ']') break;
            element += ShortToString(c);
            pos++;
         }
         StringTrimRight(element);
         StringTrimLeft(element);
         if(element != "")
         {
            ArrayResize(items, count + 1);
            items[count] = element;
            count++;
         }
      }
   }

   return count;
}

//+------------------------------------------------------------------+
//| Extract a complete JSON object { ... } starting at pos            |
//+------------------------------------------------------------------+
string ExtractObject(string &json, int startPos)
{
   int len = StringLen(json);
   if(startPos >= len) return "";

   ushort ch = StringGetCharacter(json, startPos);
   if(ch != '{') return "";

   int depth = 0;
   bool inString = false;
   string result = "";

   for(int i = startPos; i < len; i++)
   {
      ch = StringGetCharacter(json, i);
      result += ShortToString(ch);

      if(inString)
      {
         if(ch == '\\' && i + 1 < len)
         {
            i++;
            result += ShortToString(StringGetCharacter(json, i));
            continue;
         }
         if(ch == '"')
            inString = false;
         continue;
      }

      if(ch == '"')
         inString = true;
      else if(ch == '{')
         depth++;
      else if(ch == '}')
      {
         depth--;
         if(depth == 0)
            return result;
      }
   }

   return result;
}

//+------------------------------------------------------------------+
//| Extract a quoted string starting at pos (without the quotes)      |
//+------------------------------------------------------------------+
string ExtractQuotedString(string &json, int startPos)
{
   int len = StringLen(json);
   if(startPos >= len) return "";

   // Skip opening quote
   int pos = startPos + 1;
   string result = "";

   while(pos < len)
   {
      ushort ch = StringGetCharacter(json, pos);
      if(ch == '"') break;
      if(ch == '\\' && pos + 1 < len)
      {
         pos++;
         ch = StringGetCharacter(json, pos);
      }
      result += ShortToString(ch);
      pos++;
   }

   return result;
}
//+------------------------------------------------------------------+
