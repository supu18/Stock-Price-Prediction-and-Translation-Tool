File Translator Tool

Overview:

The File Translator Tool is a simple application that allows users to translate text content from various file formats. It supports translation from the following file types: .txt, .docx, .xlsx, .csv.

Features:

- Translate text content from supported file formats.
- Detect the language of the original text.
- Choose the destination language for translation.
- Save the translated text to a new file.

Requirements:

Ensure that you have Python installed on your system.

Install the required Python libraries using the following command:

pip install langdetect googletrans==4.0.0rc1 python-docx openpyxl reportlab PyPDF2

Usage:

1. Run the application by executing the file_translator.py script:

python file_translator.py

2. Click on the "Translate File" button to select a file for translation.

3. Choose the destination language from the dropdown menu.

4. Click the "Translate" button to perform the translation.

5. The translated text will be saved to a new file with the destination language appended to the original file name.

6. A success message will be displayed, and you can close the application.

Note:

- The application currently supports translation for .txt, .docx, .xlsx, .csv files.
- PDF translation is a work in progress and may have limitations.

Troubleshooting:

- If you encounter any issues, check the translation.log file for error details.

Contributions:

You are welcome to contribute to this project by opening an issue for problems or suggesting improvements. Feel free to submit a pull request.

License:

This project is licensed under the MIT License - see the LICENSE file for details.
