import xml.etree.ElementTree as ET

def parse_cranfield_documents(xml_file_path):
    """
    Parses the Cranfield collection XML file (cran.all.1400.xml) and returns a dict:
        { doc_id: document_text }
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    docs = {}

    for doc in root.findall('doc'):
        doc_id = doc.find('docno').text.strip()

        # Safely extract text from each field
        title_elem = doc.find('title')
        title = title_elem.text.strip() if title_elem is not None and title_elem.text is not None else ""

        author_elem = doc.find('author')
        author = author_elem.text.strip() if author_elem is not None and author_elem.text is not None else ""

        biblio_elem = doc.find('biblio')
        biblio = biblio_elem.text.strip() if biblio_elem is not None and biblio_elem.text is not None else ""

        text_elem = doc.find('text')
        text = text_elem.text.strip() if text_elem is not None and text_elem.text is not None else ""

        combined_text = " ".join([title, author, biblio, text])
        docs[doc_id] = combined_text

    return docs

def parse_cranfield_queries(xml_file_path):
    """
    Parses the Cranfield queries file (cran.qry.xml) and returns a dict:
        { query_id: query_text }
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    queries = {}

    for query in root.findall('top'):
        q_id = query.find('num').text.strip()
        q_text = query.find('title').text.strip()
        queries[q_id] = q_text

    return queries 