##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

Kattentemp is a project that simplifies the analysis and response generation process for USPTO Office Action letters. It offers a user-friendly interface for uploading PDFs, extracting relevant sections, and generating tailored responses. With features like chat interaction, debugging support, and customizable response formats, it caters to legal professionals seeking efficient and accurate document analysis.

##  Project Structure

```sh
└── Katten_temp/
    ├── Front_End
    │   ├── .streamlit
    │   │   ├── config.toml
    │   │   └── styles.css
    │   ├── README.md
    │   ├── app_test2.py
    │   ├── app_test_v1.py
    │   ├── requirements.txt
    │   └── utils
    │       └── helpers.py
    └── function_calls
        ├── README.md
        ├── __init__.py
        ├── generation.py
        ├── hyperlink.py
        └── retriever.py
```


###  Project Index
<details open>
	<summary><b><code>KATTEN_TEMP/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			</table>
		</blockquote>
	</details>
	<details> <!-- Front_End Submodule -->
		<summary><b>Front_End</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/app_test_v1.py'>app_test_v1.py</a></b></td>
				<td>- Facilitates interactive analysis and response generation for USPTO Office Action letters<br>- Enables users to upload PDFs, extract text, retrieve relevant sections, and generate responses based on user queries and selected sections<br>- Offers configuration options, chat interface, and download functionalities for responses in JSON or text format<br>- Supports debugging and session state management.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/app_test2.py'>app_test2.py</a></b></td>
				<td>- Facilitates a user-friendly interface for analyzing and generating responses to USPTO Office Action letters<br>- Enables users to upload PDFs, retrieve relevant sections, select key information, and generate a final response<br>- The code manages session state, displays chat messages, and handles user interactions seamlessly within a Streamlit application.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/requirements.txt'>requirements.txt</a></b></td>
				<td>- Manage core dependencies for the Streamlit application, LangChain, OpenAI, vector store, PDF processing, environment, and utility packages<br>- Include packages for regex operations, better performance, development, debugging, and ranking<br>- The file specifies required dependencies to ensure smooth functioning of the project's various components.</td>
			</tr>
			</table>
			<details>
				<summary><b>.streamlit</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/.streamlit/styles.css'>styles.css</a></b></td>
						<td>Define consistent styling for chat messages, references, message roles, selection checkboxes, status, sections, cards, subsections, summaries, and success indicators in the Streamlit application interface.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/.streamlit/config.toml'>config.toml</a></b></td>
						<td>Configures the appearance of the chat interface in the Streamlit app, defining colors for the header, background, user messages, and assistant messages.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>utils</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/Front_End/utils/helpers.py'>helpers.py</a></b></td>
						<td>- Improve data handling and processing efficiency in the project by utilizing helper functions in the Front_End/utils/helpers.py file<br>- This code enhances the overall architecture by providing reusable methods for common tasks, promoting code modularity and maintainability.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- function_calls Submodule -->
		<summary><b>function_calls</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/function_calls/retriever.py'>retriever.py</a></b></td>
				<td>- The code file `retriever.py` facilitates the retrieval of relevant sections from a Trademark Guidebook PDF by parsing subsections and creating a vector database<br>- It offers methods for retrieval using vector-based, BM25 scoring, and RFF techniques, enhancing search accuracy<br>- The code ensures error handling and provides detailed insights into the document content, aiding in efficient information retrieval.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/function_calls/generation.py'>generation.py</a></b></td>
				<td>- Facilitates generating responses based on legal sections and user input by leveraging a chat model<br>- Extracts section references from input, matches them with content, and prompts for responses<br>- Maintains chat history for context and handles follow-up questions with elaboration<br>- Enhances user experience by providing tailored responses through a structured template.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/4518wqp/Katten_temp/blob/master/function_calls/hyperlink.py'>hyperlink.py</a></b></td>
				<td>- Implements a function to search a CSV file for a specific header and return the corresponding link<br>- The function reads the CSV file, searches for the header, and retrieves the link if found<br>- It ensures the header is treated as a string for accurate matching.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with Katten_temp, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install Katten_temp using one of the following methods:

**Build from source:**

1. Clone the Katten_temp repository:
```sh
❯ git clone https://github.com/4518wqp/Katten_temp
```

2. Navigate to the project directory:
```sh
❯ cd Katten_temp
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r Front_End/requirements.txt
```




###  Usage
Run Katten_temp using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```