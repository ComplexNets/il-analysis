import pandas as pd
import re
from typing import Dict, List, Optional, Tuple

class IonicLiquidAnalyzer:
    def __init__(self):
        # Cation patterns with full names based on the dataset analysis
        self.cation_patterns = [
            # Common cations found in the dataset
            (r'(?:tri)?methylammonium', 'Ammonium'),
            (r'(?:tri)?ethylammonium', 'Ammonium'),
            (r'(?:tri)?propylammonium', 'Ammonium'),
            (r'(?:tri)?butylammonium', 'Ammonium'),
            (r'(?:tri)?pentylammonium', 'Ammonium'),
            (r'(?:tri)?hexylammonium', 'Ammonium'),
            (r'(?:tri)?heptylammonium', 'Ammonium'),
            (r'(?:tri)?octylammonium', 'Ammonium'),
            (r'(?:tri)?nonylammonium', 'Ammonium'),
            (r'(?:tri)?decylammonium', 'Ammonium'),
            (r'(?:di)?(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)ammonium', 'Ammonium'),
            (r'ammonium', 'Ammonium'),
            
            # Imidazolium cations (multiple patterns to cover all variations)
            (r'(?:1-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-)?(?:3-(?:methyl|ethyl|propyl|butyl))?imidazolium', 'Imidazolium'),
            (r'imidazolium', 'Imidazolium'),
            
            # Phosphonium cations
            (r'(?:tri)?(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)phosphonium', 'Phosphonium'),
            (r'phosphonium', 'Phosphonium'),
            
            # Pyrrolidinium cations
            (r'(?:1-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-)?(?:1-(?:methyl|ethyl|propyl|butyl))?pyrrolidinium', 'Pyrrolidinium'),
            (r'pyrrolidinium', 'Pyrrolidinium'),
            
            # Other cations
            (r'(?:1-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-)?morpholinium', 'Morpholinium'),
            (r'morpholinium', 'Morpholinium'),
            
            (r'(?:1-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-)?piperidinium', 'Piperidinium'),
            (r'piperidinium', 'Piperidinium'),
            
            (r'(?:1-(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-)?pyridinium', 'Pyridinium'),
            (r'pyridinium', 'Pyridinium'),
            
            (r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)-phospholium', 'Phospholium'),
            (r'phospholium', 'Phospholium'),
        ]
        
        # Anion patterns with full names based on the dataset analysis
        self.anion_patterns = [
            (r'tetrafluoroborate', 'Tetrafluoroborate'),
            (r'BF4', 'Tetrafluoroborate'),
            (r'bis\(trifluoromethanesulfonyl\)imide', 'Bis(trifluoromethanesulfonyl)imide'),
            (r'bis\(trifluoromethylsulfonyl\)imide', 'Bis(trifluoromethanesulfonyl)imide'),
            (r'NTf2', 'Bis(trifluoromethanesulfonyl)imide'),
            (r'acetate', 'Acetate'),
            (r'methanesulfonate', 'Methanesulfonate'),
            (r'dicyanamide', 'Dicyanamide'),
            (r'DCA', 'Dicyanamide'),
            (r'trifluoromethanesulfonate', 'Trifluoromethanesulfonate'),
            (r'OTf', 'Trifluoromethanesulfonate'),
            (r'triflate', 'Trifluoromethanesulfonate'),
            (r'formate', 'Formate'),
            (r'tosylate', 'Tosylate'),
            (r'trifluoroacetate', 'Trifluoroacetate'),
            (r'bromide', 'Bromide'),
            (r'chloride', 'Chloride'),
            (r'iodide', 'Iodide'),
            (r'hexafluorophosphate', 'Hexafluorophosphate'),
            (r'PF6', 'Hexafluorophosphate'),
        ]

        # Alkyl chain patterns with full names based on the dataset analysis
        self.alkyl_patterns = [
            # Direct mention with 1- prefix
            (r'1-methyl', 'methyl'),
            (r'1-ethyl', 'ethyl'),
            (r'1-propyl', 'propyl'),
            (r'1-butyl', 'butyl'),
            (r'1-pentyl', 'pentyl'),
            (r'1-hexyl', 'hexyl'),
            (r'1-heptyl', 'heptyl'),
            (r'1-octyl', 'octyl'),
            (r'1-nonyl', 'nonyl'),
            (r'1-decyl', 'decyl'),
            
            # Prefixes for names like ethyltrimethylammonium
            (r'^ethyl(?=tri)', 'ethyl'),
            (r'^methyl(?=tri)', 'methyl'),
            (r'^propyl(?=tri)', 'propyl'),
            (r'^butyl(?=tri)', 'butyl'),
            (r'^pentyl(?=tri)', 'pentyl'),
            (r'^hexyl(?=tri)', 'hexyl'),
            (r'^heptyl(?=tri)', 'heptyl'),
            (r'^octyl(?=tri)', 'octyl'),
            (r'^nonyl(?=tri)', 'nonyl'),
            (r'^decyl(?=tri)', 'decyl'),
            
            # Standalone prefixes
            (r'^ethyl', 'ethyl'),
            (r'^methyl', 'methyl'),
            (r'^propyl', 'propyl'),
            (r'^butyl', 'butyl'),
            (r'^pentyl', 'pentyl'),
            (r'^hexyl', 'hexyl'),
            (r'^heptyl', 'heptyl'),
            (r'^octyl', 'octyl'),
            (r'^nonyl', 'nonyl'),
            (r'^decyl', 'decyl'),
            
            # Additional patterns for trimethyl/ethyl compounds
            (r'trimethyl', 'methyl'),
            (r'triethyl', 'ethyl'),
            (r'dimethyl', 'methyl'),
            (r'diethyl', 'ethyl'),
        ]
        
        # Functional group patterns
        self.functional_group_patterns = [
            # Common oxygen-containing groups
            (r'hydroxyl?', 'Hydroxyl'),  # Match both hydroxyl and hydroxy
            (r'-?oh\b', 'Hydroxyl'),     # Match OH and -OH
            
            # Nitrogen-containing groups
            (r'amino', 'Amino'),
            (r'-?nh2\b', 'Amino'),       # Match NH2 and -NH2
            
            # Carboxyl groups
            (r'carboxyl(?:ic)?', 'Carboxyl'),  # Match carboxyl and carboxylic
            (r'-?cooh\b', 'Carboxyl'),         # Match COOH and -COOH
            
            # Carbonyl groups
            (r'carbonyl', 'Carbonyl'),
            (r'-?c=o\b', 'Carbonyl'),          # Match C=O and -C=O
            (r'-?c\(=o\)', 'Carbonyl'),        # Match C(=O) format
        ]
        
        # Compile special case patterns for faster matching
        self.special_cases = self._compile_special_cases()

    def _compile_special_cases(self) -> Dict[str, Tuple[str, str, str, Optional[str]]]:
        """Compile special case patterns for common ionic liquids."""
        # Format: name pattern -> (cation, anion, alkyl, functional_group)
        return {
            r'(?:EMIM|emim)': ('Imidazolium', None, 'ethyl', None),
            r'(?:BMIM|bmim)': ('Imidazolium', None, 'butyl', None),
            r'(?:PMIM|pmim)': ('Imidazolium', None, 'propyl', None),
            r'(?:HMIM|hmim)': ('Imidazolium', None, 'hexyl', None),
            r'(?:OMIM|omim)': ('Imidazolium', None, 'octyl', None),
            r'(?:BPy|bpy)': ('Pyridinium', None, 'butyl', None),
            r'(?:BMPyrr|bmpyrr)': ('Pyrrolidinium', None, 'butyl', None),
        }

    def extract_parts(self, name: str) -> Tuple[str, str]:
        """
        Extract cation and anion parts from an ionic liquid name.
        
        Args:
            name: The ionic liquid name
            
        Returns:
            Tuple of (cation_part, anion_part)
        """
        # Try to split by space first
        parts = name.split()
        if len(parts) > 1:
            return parts[0], parts[1]
        
        # Try to find anion at the end
        for anion_pattern, _ in self.anion_patterns:
            match = re.search(f"({anion_pattern})$", name, re.IGNORECASE)
            if match:
                anion_part = match.group(1)
                cation_part = name[:match.start()]
                return cation_part, anion_part
        
        # If no clear division, return the whole name as cation part and empty string as anion part
        return name, ""

    def match_pattern(self, text: str, patterns: List[Tuple[str, str]]) -> Optional[str]:
        """
        Match a text against a list of patterns.
        
        Args:
            text: The text to match
            patterns: List of (pattern, result) tuples
            
        Returns:
            Matching result or None
        """
        if not text:
            return None
            
        text = text.lower()  # Convert to lowercase for case-insensitive matching
        for pattern, result in patterns:
            if re.search(pattern, text, re.IGNORECASE):  # Use re.IGNORECASE for case-insensitive matching
                return result
        return None
    
    def check_special_cases(self, name: str) -> Optional[Dict]:
        """
        Check for special case ionic liquids.
        
        Args:
            name: The ionic liquid name
            
        Returns:
            Dictionary of components or None
        """
        for pattern, (cation, anion, alkyl, functional_group) in self.special_cases.items():
            if re.search(pattern, name, re.IGNORECASE):
                # Only return components that are specified in the special case
                # For unspecified components, we'll use the regular matching process
                components = {'Name': name}
                if cation:
                    components['cation'] = cation
                if anion:
                    components['anion'] = anion
                if alkyl:
                    components['alkyl_chains'] = alkyl
                if functional_group:
                    components['functional_group'] = functional_group
                return components
        return None

    def analyze_ionic_liquid(self, name: str) -> Dict:
        """
        Analyze an ionic liquid name and break it down into components.
        
        Args:
            name: The ionic liquid name
            
        Returns:
            Dictionary of identified components
        """
        if not name or name.lower() == 'nan':
            return {
                'Name': name,
                'cation': None,
                'anion': None,
                'alkyl_chains': None,
                'functional_group': None
            }
        
        # Check for special cases first
        special_case = self.check_special_cases(name)
        if special_case:
            # We still need to fill in any missing components
            cation_part, anion_part = self.extract_parts(name)
            
            if 'cation' not in special_case:
                special_case['cation'] = self.match_pattern(cation_part, self.cation_patterns)
            
            if 'anion' not in special_case:
                special_case['anion'] = self.match_pattern(anion_part, self.anion_patterns)
                # If no anion was found in anion_part, try the whole name
                if not special_case['anion']:
                    special_case['anion'] = self.match_pattern(name, self.anion_patterns)
            
            if 'alkyl_chains' not in special_case:
                special_case['alkyl_chains'] = self.match_pattern(cation_part, self.alkyl_patterns)
            
            if 'functional_group' not in special_case:
                special_case['functional_group'] = self.match_pattern(name, self.functional_group_patterns)
            
            return special_case
        
        # Extract parts
        cation_part, anion_part = self.extract_parts(name)
        
        # Match each component
        cation = self.match_pattern(cation_part, self.cation_patterns)
        anion = self.match_pattern(anion_part, self.anion_patterns)
        
        # If no anion was found in anion_part, try the whole name
        if not anion:
            anion = self.match_pattern(name, self.anion_patterns)
        
        # Special handling for alkyl chains
        alkyl = self.match_pattern(cation_part, self.alkyl_patterns)
        
        # Find functional groups
        functional_group = self.match_pattern(name, self.functional_group_patterns)
        
        # Special case for ammonium compounds with format like "ethyltrimethylammonium chloride"
        if cation == 'Ammonium' and not alkyl:
            ammonium_prefix = re.search(r'^(methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)', 
                                        name, re.IGNORECASE)
            if ammonium_prefix:
                alkyl = ammonium_prefix.group(1).lower()
        
        # Extra handling for difficult cases
        if not cation and ('ammonium' in name.lower() or 'phosphonium' in name.lower()):
            if 'ammonium' in name.lower():
                cation = 'Ammonium'
            elif 'phosphonium' in name.lower():
                cation = 'Phosphonium'
            
            # Try to extract alkyl from trialkylammonium format
            alkyl_match = re.search(r'(methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl)(?=tri)', 
                                   name, re.IGNORECASE)
            if alkyl_match:
                alkyl = alkyl_match.group(1).lower()
        
        return {
            'Name': name,
            'cation': cation,
            'anion': anion,
            'alkyl_chains': alkyl,
            'functional_group': functional_group
        }

def process_excel_file(input_file: str, output_file: str, verbose: bool = False):
    """
    Process an Excel file with ionic liquid names and save the analyzed results.
    
    Args:
        input_file (str): Path to the input Excel file containing ionic liquid names
        output_file (str): Path where the analyzed results will be saved as Excel file
        verbose (bool, optional): Whether to print progress information. Defaults to False.
    """
    try:
        # Read the input file
        df = pd.read_excel(input_file)
        
        # Create analyzer
        analyzer = IonicLiquidAnalyzer()
        
        # Process each row
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if verbose and idx % 10 == 0:
                print(f"Processing row {idx+1}/{total_rows}")
                
            name = str(row['Name'])
            analysis = analyzer.analyze_ionic_liquid(name)
            
            # Add all original property columns to the analysis
            for col in df.columns:
                if col != 'Name':  # Skip Name since it's already processed
                    analysis[col] = row[col]
            
            results.append(analysis)
            
        # Create result dataframe
        result_df = pd.DataFrame(results)
        
        # Ensure the functional_group column is included and not empty
        if 'functional_group' in result_df.columns:
            # Check if all values are None or empty
            if result_df['functional_group'].isna().all() or (result_df['functional_group'] == '').all():
                print("Warning: All functional_group values are empty. Check the analyzer.functional_group_patterns.")
        else:
            print("Warning: functional_group column is missing from the results.")
            
        # Save to output file
        result_df.to_excel(output_file, index=False)
        
        if verbose:
            print(f"Analysis complete. Results saved to {output_file}")
            
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_file} is empty")
        raise
    except FileNotFoundError:
        print(f"Error: Could not find the file {input_file}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    process_excel_file("il_prop_in.xlsx", "il_prop_analyzed.xlsx", verbose=True)