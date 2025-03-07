# LLM-Powered-Questionnaire-Generator

---

**Developed by:** Jai Ansh Singh Bindra, Anubhav Choudhery

---

## Overview

The **LLM-Powered Edu-Video Questionnaire Generator** is a complete pipeline that takes a **YouTube video link** as input and returns a **concise, diverse, and thought-provoking questionnaire** as output. This project is designed to enhance the learning experience by providing engaging questions that promote **active recall and reassessment** of knowledge gained from the video.

Our pipeline leverages a combination of **state-of-the-art LLMs (Large Language Models)** and **other optimization techniques** to ensure efficient processing. By using **strategic chunking, segmentation, and GPU-CPU optimization**, we have successfully reduced the pipeline's run-time from approximately **1 hour to just 5-7 minutes (a 92% improvement!)**. 

We implemented **adaptive chunking and segmentation techniques** for handling large inputs efficiently. Instead of processing the entire transcription in a single pass, the text is divided into manageable segments, ensuring high-quality summarization and questionnaire generation while maintaining speed. Additionally, we strategically leverage a **GPU-CPU hybrid approach**, where GPU acceleration is used for model inference while CPU processing handles text manipulations, resulting in a **consistent run-time of 5-8 minutes**, even for videos as long as 46 minutes.

Currently, this project supports **YouTube videos** and is intended for **educational, learning, and skill-development purposes**. The generated questionnaire is provided in a **PDF format**, making it easy to use and store for future reference.

## Features

- **Automated Processing**: Extracts audio, transcribes, summarizes, and generates insightful questions.
- **Optimized Performance**: Uses **GPU-CPU hybrid processing** to maintain a uniform run-time between **5-8 minutes**, even for long videos (~46 minutes tested).
- **Adaptive Chunking & Segmentation**: Ensures high-quality summaries and questionnaire generation with minimal processing time.
- **Kaggle Notebook Compatibility**: The pipeline is optimized for **Kaggle notebooks**, making it easy to run without additional setup.
- **PDF Output**: The questionnaire is saved as a **PDF** for convenience and future reference.

## Installation & Dependencies

To use this pipeline, you'll need the following dependencies installed:

### Required Python Libraries
```bash
pip install -r requirements.txt
```
The **requirements.txt** file includes:
- `yt-dlp` (for YouTube video/audio extraction)
- `whisper` (for transcription)
- `transformers` (for LLM-based summarization and question generation)
- `pdfkit` & `wkhtmltopdf` (for generating PDFs)
- `torch` (for model acceleration)
- `opencv-python` (if video preprocessing is required)

### Running on Kaggle Notebooks
We recommend running this pipeline on **Kaggle notebooks**, which provide **free GPUs**. The repository contains ready-to-use Kaggle-compatible code. Simply upload the notebook and run the steps as instructed.

## Usage

1. **Clone the Repository**
```bash
git clone https://github.com/AnubhavChoudhery/LLM-Powered-Questionnaire-Generator.git
cd LLM-Powered-Questionnaire-Generator
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Pipeline**
```bash
python main.py --video_url "<YouTube Video URL>"
```

4. **Retrieve the Output**
- The generated **questionnaire** will be saved as a **PDF file** in the `output/` directory.

## How It Works

1. **Video Processing**
   - Extracts audio from the YouTube video using `yt-dlp`.
   - Uses OpenAI Whisper for accurate transcription.

2. **Summarization & Question Generation**
   - Applies **chunking/segmentation** for optimal LLM performance.
   - Uses **transformers-based models** for summarization.
   - Generates diverse and thought-provoking questions.

3. **Performance Optimization**
   - Implements **adaptive segmentation** to ensure efficient processing of large transcripts.
   - Uses a **GPU-CPU hybrid approach**, where GPU handles model inference and CPU processes text operations for improved efficiency.
   - Maintains a **uniform processing time** (~5-8 minutes), even for longer videos (~46 minutes tested).

4. **Output Generation**
   - Formats questions into a structured PDF.
   - Uses `pdfkit` and `wkhtmltopdf` for professional styling.

## Future Plans
- **Expand Beyond YouTube**: We aim to make our tool work with local video files and other video platforms as well (expand beyond just youtube videos).
- **Deploy on Hugging Face Spaces or something similiar**: To allow the users to generate questionnaires on an online platform instead of needing to have the environment to run the pipeline themselves.
- **Enhanced Question Diversity**: Bring in even better question variety!

## Contributing
We welcome contributions! Feel free to **open issues, submit PRs**, or reach out with suggestions!
[**Contact:** [Jaianshofficial26@gmail.com](mailto:jaianshofficial26@gmail.com), [Anubhavchoudhery95@gmail.com](mailto:anubhavchoudhery95@gmail.com)]

## License
This project is open-source and available under the **MIT License**.
