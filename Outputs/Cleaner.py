#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

def renumber_queries():
    # Modify these paths as needed
    input_file = "cran.qry.old.xml"
    output_file = "cran.qry.xml"

    # Parse the XML
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find all <top> elements
    top_elements = root.findall('top')

    # Renumber each <num> tag sequentially
    for i, top in enumerate(top_elements, start=1):
        num_tag = top.find('num')
        if num_tag is not None:
            num_tag.text = str(i)

    # Write the modified XML to output
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Successfully renumbered queries in {input_file} -> {output_file}")

if __name__ == "__main__":
    renumber_queries()
