@echo off
echo ========================================
echo    TRANSFORMER-C DOCUMENTATION UPDATER
echo ========================================
echo.

echo [1/7] Creating .gitignore...
echo # Build outputs > .gitignore
echo bin/ >> .gitignore
echo obj/ >> .gitignore
echo *.exe >> .gitignore
echo *.o >> .gitignore
echo *.out >> .gitignore
echo a.out >> .gitignore
echo. >> .gitignore
echo # Models and training data >> .gitignore
echo saved_models/ >> .gitignore
echo *.model >> .gitignore
echo *.weights >> .gitignore
echo. >> .gitignore
echo # Analysis outputs >> .gitignore
echo analysis_reports/ >> .gitignore
echo *.log >> .gitignore
echo *.txt >> .gitignore
echo novel_errors.txt >> .gitignore
echo my_terminal_output.log >> .gitignore
echo OK - Created .gitignore

echo.
echo [2/7] Creating README.md...
echo # Transformer-C > README.md
echo. >> README.md
echo A complete transformer neural network implementation in C with interactive training. >> README.md
echo. >> README.md
echo ## Features >> README.md
echo - Transformer architecture with 12 attention heads >> README.md
echo - Interactive REPL for training and testing >> README.md
echo - Model saving and loading >> README.md
echo - Text generation and prediction >> README.md
echo - Built-in analysis tools >> README.md
echo. >> README.md
echo ## Quick Start >> README.md
echo. >> README.md
echo ### Build the project >> README.md
echo \`\`\`bash >> README.md
echo make >> README.md
echo \`\`\` >> README.md
echo. >> README.md
echo ### Run the REPL >> README.md
echo \`\`\`bash >> README.md
echo .\bin\transformer_c.exe >> README.md
echo \`\`\` >> README.md
echo. >> README.md
echo ## REPL Commands >> README.md
echo - load ^<model^>      - Load saved model >> README.md
echo - train ^<epochs^>    - Train model >> README.md
echo - predict ^<text^>    - Predict next word >> README.md
echo - generate ^<length^> - Generate text >> README.md
echo - save ^<name^>       - Save model >> README.md
echo - info              - Show model info >> README.md
echo - weights           - Show weights >> README.md
echo - list              - List models >> README.md
echo - help              - Show help >> README.md
echo - exit              - Exit REPL >> README.md
echo. >> README.md
echo ## Model Architecture >> README.md
echo - Attention Heads: 12 (4 layers) >> README.md
echo - Embedding Dimension: 2 >> README.md
echo - Matrix Size: 2x2 per head >> README.md
echo - Semi-final layer: 33,280 parameters >> README.md
echo - Final layer: 130 parameters >> README.md
echo - Max sequence length: 512 tokens >> README.md
echo. >> README.md
echo ## Building >> README.md
echo \`\`\`bash >> README.md
echo make clean >> README.md
echo make >> README.md
echo make test >> README.md
echo \`\`\` >> README.md
echo. >> README.md
echo ## License >> README.md
echo MIT License >> README.md
echo OK - Created README.md

echo.
echo [3/7] Creating docs/index.md...
mkdir docs 2>nul
echo # Transformer-C Documentation > docs/index.md
echo. >> docs/index.md
echo ## Architecture >> docs/index.md
echo. >> docs/index.md
echo ### Model Components >> docs/index.md
echo 1. Self-Attention Layer (12 heads) >> docs/index.md
echo 2. Feed Forward Network >> docs/index.md
echo 3. Training System >> docs/index.md
echo 4. Model Management >> docs/index.md
echo. >> docs/index.md
echo ### File Structure >> docs/index.md
echo - src/main.c - REPL interface >> docs/index.md
echo - src/self_attention_layer.c - Attention mechanism >> docs/index.md
echo - src/backpropagation.c - Training >> docs/index.md
echo - include/transformer_block.h - Main architecture >> docs/index.md
echo. >> docs/index.md
echo ## Usage Examples >> docs/index.md
echo. >> docs/index.md
echo ### Training >> docs/index.md
echo \`\`\`bash >> docs/index.md
echo transformer-c^> load trained_model_v1 >> docs/index.md
echo transformer-c^> train 10 >> docs/index.md
echo transformer-c^> save improved_model >> docs/index.md
echo \`\`\` >> docs/index.md
echo. >> docs/index.md
echo ### Generation >> docs/index.md
echo \`\`\`bash >> docs/index.md
echo transformer-c^> generate 50 >> docs/index.md
echo \`\`\` >> docs/index.md
echo. >> docs/index.md
echo ## Troubleshooting >> docs/index.md
echo. >> docs/index.md
echo ### Common Issues >> docs/index.md
echo 1. "Model not loaded" - Load or train a model first >> docs/index.md
echo 2. Low accuracy - Train for more epochs >> docs/index.md
echo 3. Memory issues - Models are ~270KB each >> docs/index.md
echo OK - Created docs/index.md

echo.
echo [4/7] Creating Makefile...
echo # Transformer-C Makefile > Makefile
echo CC = gcc >> Makefile
echo CFLAGS = -Wall -Wextra -O2 -std=c99 -I./include >> Makefile
echo TARGET = bin/transformer_c.exe >> Makefile
echo. >> Makefile
echo # Source files >> Makefile
echo SRC = $(wildcard src/*.c) >> Makefile
echo OBJ = $(patsubst src/%.c, obj/%.o, $(SRC)) >> Makefile
echo. >> Makefile
echo # Default target >> Makefile
echo all: $(TARGET) >> Makefile
echo. >> Makefile
echo # Link executable >> Makefile
echo $(TARGET): $(OBJ) >> Makefile
echo     @mkdir -p bin >> Makefile
echo     $(CC) $(CFLAGS) -o $@ $^ >> Makefile
echo. >> Makefile
echo # Compile objects >> Makefile
echo obj/%.o: src/%.c >> Makefile
echo     @mkdir -p obj >> Makefile
echo     $(CC) $(CFLAGS) -c $< -o $@ >> Makefile
echo. >> Makefile
echo # Clean build >> Makefile
echo clean: >> Makefile
echo     rm -rf obj bin/*.exe >> Makefile
echo. >> Makefile
echo # Run tests >> Makefile
echo test: $(TARGET) >> Makefile
echo     ./$(TARGET) --test >> Makefile
echo. >> Makefile
echo .PHONY: all clean test >> Makefile
echo OK - Created Makefile

echo.
echo [5/7] Creating LICENSE file...
echo MIT License > LICENSE
echo. >> LICENSE
echo Copyright (c) 2024 Your Name >> LICENSE
echo. >> LICENSE
echo Permission is hereby granted, free of charge, to any person obtaining a copy >> LICENSE
echo of this software and associated documentation files (the "Software"), to deal >> LICENSE
echo in the Software without restriction, including without limitation the rights >> LICENSE
echo to use, copy, modify, merge, publish, distribute, sublicense, and/or sell >> LICENSE
echo copies of the Software, and to permit persons to whom the Software is >> LICENSE
echo furnished to do so, subject to the following conditions: >> LICENSE
echo. >> LICENSE
echo The above copyright notice and this permission notice shall be included in all >> LICENSE
echo copies or substantial portions of the Software. >> LICENSE
echo. >> LICENSE
echo THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR >> LICENSE
echo IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, >> LICENSE
echo FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE >> LICENSE
echo AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER >> LICENSE
echo LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, >> LICENSE
echo OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE >> LICENSE
echo SOFTWARE. >> LICENSE
echo OK - Created LICENSE

echo.
echo [6/7] Creating git commit script...
echo @echo off > commit.bat
echo git add . >> commit.bat
echo git commit -m "Update documentation and project structure" >> commit.bat
echo git push >> commit.bat
echo OK - Created commit.bat

echo.
echo [7/7] Cleaning up...
mkdir tools 2>nul
if exist analyzer.exe move analyzer.exe tools\ >nul
if exist project_analyzer.c move project_analyzer.c tools\ >nul
echo OK - Organized tools

echo.
echo ========================================
echo    DOCUMENTATION UPDATE COMPLETE!
echo ========================================
echo.
echo Files created/updated:
echo   .gitignore
echo   README.md
echo   docs/index.md
echo   Makefile
echo   LICENSE
echo   commit.bat
echo.
echo To commit to GitHub:
echo   1. Run: commit.bat
echo   2. Or manually:
echo      git add .
echo      git commit -m "Update docs"
echo      git push
echo.
pause