#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #define PATH_SEP '\\'
    #define IS_DIR(attr) ((attr) & FILE_ATTRIBUTE_DIRECTORY)
#else
    #include <dirent.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #define PATH_SEP '/'
#endif

#define MAX_PATH_LEN 512
#define MAX_LINE_LEN 1024
#define MAX_FILES 5000
#define MAX_FUNCTIONS 1000

typedef struct {
    char name[256];
    char path[MAX_PATH_LEN];
    long size;
    int lines;
    int is_c;
    int is_h;
} FileInfo;

typedef struct {
    FileInfo* files;
    int count;
    int capacity;
    long total_lines;
    long c_lines;
    long h_lines;
    int c_files;
    int h_files;
    long total_size;
} FileList;

// Safe string copy
void safe_strcpy(char* dest, const char* src, size_t max_len) {
    if (dest && src && max_len > 0) {
        strncpy(dest, src, max_len - 1);
        dest[max_len - 1] = '\0';
    }
}

// Check if file is C source
int is_c_file(const char* name) {
    const char* dot = strrchr(name, '.');
    if (!dot) return 0;
    
    if (strcmp(dot, ".c") == 0) return 1;
    if (strcmp(dot, ".cpp") == 0) return 1;
    if (strcmp(dot, ".cc") == 0) return 1;
    return 0;
}

// Check if file is header
int is_h_file(const char* name) {
    const char* dot = strrchr(name, '.');
    if (!dot) return 0;
    
    if (strcmp(dot, ".h") == 0) return 1;
    if (strcmp(dot, ".hpp") == 0) return 1;
    return 0;
}

// Count lines in file safely
int count_file_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;
    
    int lines = 0;
    char buffer[MAX_LINE_LEN];
    
    while (fgets(buffer, sizeof(buffer), file)) {
        lines++;
    }
    
    fclose(file);
    return lines;
}

#ifdef _WIN32
// Windows directory traversal
void scan_directory_win(const char* base_path, FileList* list) {
    char search_path[MAX_PATH_LEN];
    snprintf(search_path, sizeof(search_path), "%s\\*", base_path);
    
    WIN32_FIND_DATAA find_data;
    HANDLE hFind = FindFirstFileA(search_path, &find_data);
    
    if (hFind == INVALID_HANDLE_VALUE) {
        return;
    }
    
    do {
        // Skip . and ..
        if (strcmp(find_data.cFileName, ".") == 0 || 
            strcmp(find_data.cFileName, "..") == 0) {
            continue;
        }
        
        char full_path[MAX_PATH_LEN];
        snprintf(full_path, sizeof(full_path), "%s\\%s", base_path, find_data.cFileName);
        
        if (IS_DIR(find_data.dwFileAttributes)) {
            // Recursively scan directory
            scan_directory_win(full_path, list);
        } else {
            // Add file to list if we have space
            if (list->count >= list->capacity) {
                // Increase capacity
                list->capacity *= 2;
                list->files = realloc(list->files, list->capacity * sizeof(FileInfo));
                if (!list->files) {
                    printf("ERROR: Memory allocation failed!\n");
                    exit(1);
                }
            }
            
            FileInfo* file = &list->files[list->count];
            
            // Initialize file info
            memset(file, 0, sizeof(FileInfo));
            safe_strcpy(file->name, find_data.cFileName, sizeof(file->name));
            safe_strcpy(file->path, full_path, sizeof(file->path));
            
            // Get file size
            file->size = ((long long)find_data.nFileSizeHigh << 32) | find_data.nFileSizeLow;
            
            // Check file type
            file->is_c = is_c_file(find_data.cFileName);
            file->is_h = is_h_file(find_data.cFileName);
            
            // Count lines
            file->lines = count_file_lines(full_path);
            
            // Update totals
            list->total_size += file->size;
            list->total_lines += file->lines;
            
            if (file->is_c) {
                list->c_files++;
                list->c_lines += file->lines;
            } else if (file->is_h) {
                list->h_files++;
                list->h_lines += file->lines;
            }
            
            list->count++;
            
            // Show progress every 100 files
            if (list->count % 100 == 0) {
                printf("  Found %d files...\r", list->count);
                fflush(stdout);
            }
        }
    } while (FindNextFileA(hFind, &find_data) != 0);
    
    FindClose(hFind);
}
#else
// Linux/Mac directory traversal
void scan_directory_unix(const char* base_path, FileList* list) {
    DIR* dir = opendir(base_path);
    if (!dir) return;
    
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || 
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        char full_path[MAX_PATH_LEN];
        snprintf(full_path, sizeof(full_path), "%s/%s", base_path, entry->d_name);
        
        struct stat statbuf;
        if (stat(full_path, &statbuf) != 0) {
            continue;
        }
        
        if (S_ISDIR(statbuf.st_mode)) {
            // Recursively scan directory
            scan_directory_unix(full_path, list);
        } else if (S_ISREG(statbuf.st_mode)) {
            // Add file to list if we have space
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                list->files = realloc(list->files, list->capacity * sizeof(FileInfo));
                if (!list->files) {
                    printf("ERROR: Memory allocation failed!\n");
                    exit(1);
                }
            }
            
            FileInfo* file = &list->files[list->count];
            
            // Initialize file info
            memset(file, 0, sizeof(FileInfo));
            safe_strcpy(file->name, entry->d_name, sizeof(file->name));
            safe_strcpy(file->path, full_path, sizeof(file->path));
            
            // Get file size
            file->size = statbuf.st_size;
            
            // Check file type
            file->is_c = is_c_file(entry->d_name);
            file->is_h = is_h_file(entry->d_name);
            
            // Count lines
            file->lines = count_file_lines(full_path);
            
            // Update totals
            list->total_size += file->size;
            list->total_lines += file->lines;
            
            if (file->is_c) {
                list->c_files++;
                list->c_lines += file->lines;
            } else if (file->is_h) {
                list->h_files++;
                list->h_lines += file->lines;
            }
            
            list->count++;
            
            // Show progress every 100 files
            if (list->count % 100 == 0) {
                printf("  Found %d files...\r", list->count);
                fflush(stdout);
            }
        }
    }
    
    closedir(dir);
}
#endif

// Initialize file list
void init_file_list(FileList* list) {
    list->capacity = 100;
    list->files = malloc(list->capacity * sizeof(FileInfo));
    if (!list->files) {
        printf("ERROR: Initial memory allocation failed!\n");
        exit(1);
    }
    list->count = 0;
    list->total_lines = 0;
    list->c_lines = 0;
    list->h_lines = 0;
    list->c_files = 0;
    list->h_files = 0;
    list->total_size = 0;
}

// Free file list
void free_file_list(FileList* list) {
    if (list->files) {
        free(list->files);
        list->files = NULL;
    }
}

// Simple analysis - just find backprop/transformer related files
void analyze_transformer_files(FileList* list) {
    printf("\n\n=== TRANSFORMER SPECIFIC ANALYSIS ===\n\n");
    
    int backprop_files = 0;
    int attention_files = 0;
    int layer_files = 0;
    int transformer_files = 0;
    
    for (int i = 0; i < list->count; i++) {
        FileInfo* file = &list->files[i];
        char lower_name[256];
        
        // Convert to lowercase for case-insensitive search
        strcpy(lower_name, file->name);
        for (char* p = lower_name; *p; p++) *p = tolower(*p);
        
        // Check for transformer-related terms
        if (strstr(lower_name, "backprop") || 
            strstr(lower_name, "gradient") ||
            strstr(file->path, "backprop") ||
            strstr(file->path, "gradient")) {
            printf("Backprop file: %s (%d lines)\n", file->name, file->lines);
            backprop_files++;
        }
        
        if (strstr(lower_name, "attention") || 
            strstr(file->path, "attention")) {
            printf("Attention file: %s (%d lines)\n", file->name, file->lines);
            attention_files++;
        }
        
        if (strstr(lower_name, "layer") || 
            strstr(lower_name, "norm") ||
            strstr(file->path, "layer") ||
            strstr(file->path, "norm")) {
            layer_files++;
        }
        
        if (strstr(lower_name, "transformer") || 
            strstr(file->path, "transformer")) {
            printf("Transformer file: %s (%d lines)\n", file->name, file->lines);
            transformer_files++;
        }
    }
    
    printf("\nSummary:\n");
    printf("  Backprop/gradient files: %d\n", backprop_files);
    printf("  Attention files: %d\n", attention_files);
    printf("  Layer/norm files: %d\n", layer_files);
    printf("  Transformer files: %d\n", transformer_files);
    
    if (backprop_files == 0) {
        printf("\n⚠️  WARNING: No backpropagation files found!\n");
        printf("   Expected files like: backprop.c, gradient.c, train.c\n");
    }
    
    if (attention_files == 0) {
        printf("\n⚠️  WARNING: No attention mechanism files found!\n");
        printf("   Expected files like: attention.c, mha.c (multi-head attention)\n");
    }
}

// Generate simple report
void generate_simple_report(FileList* list, const char* report_name) {
    FILE* report = fopen(report_name, "w");
    if (!report) {
        printf("ERROR: Could not create report file!\n");
        return;
    }
    
    fprintf(report, "# Transformer-C Project Analysis\n\n");
    fprintf(report, "Generated: %s\n\n", __DATE__);
    
    fprintf(report, "## Summary\n\n");
    fprintf(report, "- Total files: %d\n", list->count);
    fprintf(report, "- C source files: %d (%ld lines)\n", list->c_files, list->c_lines);
    fprintf(report, "- Header files: %d (%ld lines)\n", list->h_files, list->h_lines);
    fprintf(report, "- Other files: %d (%ld lines)\n", 
            list->count - list->c_files - list->h_files,
            list->total_lines - list->c_lines - list->h_lines);
    fprintf(report, "- Total size: %.2f MB\n\n", 
            list->total_size / (1024.0 * 1024.0));
    
    // List C files
    fprintf(report, "## C Source Files\n\n");
    for (int i = 0; i < list->count; i++) {
        if (list->files[i].is_c) {
            fprintf(report, "- %s (%d lines, %.1f KB)\n", 
                    list->files[i].name,
                    list->files[i].lines,
                    list->files[i].size / 1024.0);
        }
    }
    
    // List header files
    fprintf(report, "\n## Header Files\n\n");
    for (int i = 0; i < list->count; i++) {
        if (list->files[i].is_h) {
            fprintf(report, "- %s (%d lines, %.1f KB)\n", 
                    list->files[i].name,
                    list->files[i].lines,
                    list->files[i].size / 1024.0);
        }
    }
    
    // List largest files
    fprintf(report, "\n## Largest Files (Top 10)\n\n");
    
    // Simple bubble sort by size
    for (int i = 0; i < list->count - 1; i++) {
        for (int j = 0; j < list->count - i - 1; j++) {
            if (list->files[j].size < list->files[j + 1].size) {
                FileInfo temp = list->files[j];
                list->files[j] = list->files[j + 1];
                list->files[j + 1] = temp;
            }
        }
    }
    
    for (int i = 0; i < 10 && i < list->count; i++) {
        const char* type = "Other";
        if (list->files[i].is_c) type = "C";
        else if (list->files[i].is_h) type = "H";
        
        fprintf(report, "%d. %s [%s] - %d lines, %.1f KB\n",
                i + 1,
                list->files[i].name,
                type,
                list->files[i].lines,
                list->files[i].size / 1024.0);
    }
    
    fprintf(report, "\n## Recommendations\n\n");
    fprintf(report, "1. Files over 1000 lines should be considered for splitting\n");
    fprintf(report, "2. Ensure all C files have corresponding headers\n");
    fprintf(report, "3. Check for missing backpropagation implementations\n");
    fprintf(report, "4. Verify transformer components are properly modularized\n");
    
    fclose(report);
    printf("\nReport saved to: %s\n", report_name);
}

int main(int argc, char* argv[]) {
    printf("=========================================\n");
    printf("     TRANSFORMER-C SIMPLE ANALYZER\n");
    printf("=========================================\n\n");
    
    char path[MAX_PATH_LEN] = ".";
    if (argc > 1) {
        safe_strcpy(path, argv[1], sizeof(path));
    }
    
    printf("Analyzing: %s\n\n", path);
    
    FileList file_list;
    init_file_list(&file_list);
    
    printf("Scanning directory...\n");
    
    clock_t start = clock();
    
#ifdef _WIN32
    scan_directory_win(path, &file_list);
#else
    scan_directory_unix(path, &file_list);
#endif
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n\nScan complete!\n");
    printf("==============\n\n");
    
    printf("Files found: %d\n", file_list.count);
    printf("C files: %d\n", file_list.c_files);
    printf("Header files: %d\n", file_list.h_files);
    printf("Other files: %d\n", file_list.count - file_list.c_files - file_list.h_files);
    printf("Total lines: %ld\n", file_list.total_lines);
    printf("Scan time: %.2f seconds\n\n", elapsed);
    
    if (file_list.count == 0) {
        printf("No files found! Check the path.\n");
        free_file_list(&file_list);
        return 1;
    }
    
    // Show largest C file
    FileInfo* largest_c = NULL;
    for (int i = 0; i < file_list.count; i++) {
        if (file_list.files[i].is_c) {
            if (!largest_c || file_list.files[i].size > largest_c->size) {
                largest_c = &file_list.files[i];
            }
        }
    }
    
    if (largest_c) {
        printf("Largest C file: %s (%.1f KB, %d lines)\n", 
               largest_c->name, 
               largest_c->size / 1024.0,
               largest_c->lines);
    }
    
    // Analyze transformer-specific files
    analyze_transformer_files(&file_list);
    
    // Generate report
    char report_name[MAX_PATH_LEN];
    snprintf(report_name, sizeof(report_name), "transformer_analysis_%ld.md", (long)time(NULL));
    generate_simple_report(&file_list, report_name);
    
    // Clean up
    free_file_list(&file_list);
    
    printf("\nDone! Press Enter to exit...\n");
    getchar();
    
    return 0;
}