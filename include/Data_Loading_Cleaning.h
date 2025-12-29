#ifndef DATA_LOADING_CLEANING_H
#define DATA_LOADING_CLEANING_H

char *readFileToString(const char *filename);
char **SplitSentences(char *text);
char* Cleaned_Text(const char* raw_text);  // Changed to const char*

#endif
