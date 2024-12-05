import pandas as pd

def find_link(file_path, header_name):
    """
    Searches through a CSV file for a specific header and returns the corresponding link.
    
    :param file_path: Path to the CSV file
    :param header_name: The header to search for (e.g., 1201, 1202)
    :return: The corresponding link if the header is found, otherwise a message
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure header_name is treated as a string to match CSV values
    header_name = str(header_name)
    
    # Search for the header and get the corresponding link
    result = df.loc[df['header'] == header_name, 'link']
    
    if not result.empty:
        return result.iloc[0]
    else:
        return f"No link found for header: {header_name}"

# Example usage
file_path = "your_file.csv"  # Replace with your CSV file path
header_name = 1201.01          # Replace with the header you want to search for

link = find_link(file_path, header_name)
print(link)
