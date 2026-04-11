class Surface:
    def __init__(self, name):
        """
        Initialize a Surface object with a given name.
        """
        self.name = name

    def __setattr__(self, key, value):
        """
        Override attribute setting to clean up specific values like CURV, DIAM, DISZ and PARM.
        """
        if key in ["CURV", "DISZ", "CONI"]:  # Handle CURV and DISZ specifically
            value = self._parse_first_float(value)
        elif key == "DIAM":  # Handle DIAM specifically, multiplying by 2
            value = self._parse_first_float(value) * 2
        elif key.startswith("PARM"):  # Handle PARM specifically
            param_index, param_value = self._parse_parm(value)
            key = f"PARM{param_index}"
            value = param_value
        super().__setattr__(key, value)

    @staticmethod
    def _parse_first_float(value):
        """
        Extract and return the first float value from a string.
        """
        try:
            return float(value.split()[0])  # Extract the first float
        except (ValueError, IndexError):
            return value  # Return the raw value if parsing fails
    
    @staticmethod
    def _parse_parm(value):
        """
        Parse a PARM line into its index and value.
        """
        try:
            parts = value.split()
            param_index = int(parts[0])  # Extract the PARM index
            param_value = float(parts[1])  # Extract the PARM value
            return param_index, param_value
        except (ValueError, IndexError):
            raise ValueError(f"Invalid PARM format: {value}")

    def __repr__(self):
        """
        Represent the Surface object with its name and attributes.
        """
        return f"Surface(name={self.name}, attributes={self.__dict__})"

class SystemDetails:
    def __init__(self):
        """
        Initialize the SystemDetails object with default attributes.
        """
        self.ENPD = None
        self.UNIT = None
        self.WAVM = []
        self.FEF = []

    def __repr__(self):
        """
        Represent the SystemDetails object with its attributes.
        """
        return f"SystemDetails(ENPD={self.ENPD}, UNIT={self.UNIT}, WAVM={self.WAVM}, FEFD={self.FEF})"


class ZemaxFileParser:
    def __init__(self, file_path):
        """
        Initialize the parser with the path to the Zemax .zmx file.
        Automatically reads the file and parses all relevant data.
        """
        self.file_path = file_path
        self.file_content = None
        self.surfaces = {}
        self.system_details = SystemDetails()
        self.STOP = None  # Attribute to hold the STOP surface
        
        # Automatically perform all parsing steps
        self._initialize_parser()

    def _initialize_parser(self):
        """
        Perform all necessary parsing steps during initialization.
        """
        self.read_file()
        self.parse_surfaces()
        self.identify_stop_surface()
        self.parse_system_details()

    def read_file(self):
        """
        Reads the file with UTF-16 encoding and stores the content as text.
        """
        with open(self.file_path, 'r', encoding='utf-16') as file:
            self.file_content = file.readlines()

    def parse_surfaces(self):
        """
        Parses the file content to extract surface information and create Surface objects.
        """
        if not self.file_content:
            raise ValueError("File content is empty. Did you call read_file()?")

        current_surface = None
        current_surface_number = None

        for line in self.file_content:
            line = line.strip()

            if line.startswith("SURF"):  # Start of a new surface
                if current_surface:
                    # Determine the name: use COMM if present, otherwise use the surface number
                    self._store_surface(current_surface, current_surface_number)

                # Initialize a new surface
                current_surface_number = line.split(" ", 1)[-1]
                current_surface = Surface(name=current_surface_number)
            elif current_surface and line:  # Parse details of the current surface
                key_value = line.split(" ", 1)
                if len(key_value) == 2:  # Key-value pair
                    key, value = key_value
                    setattr(current_surface, key, value)
                elif len(key_value) == 1:  # Key only (e.g., "STOP")
                    key = key_value[0]
                    setattr(current_surface, key, True)  # Assign True for standalone keys like STOP

        if current_surface:
            # Store the last surface
            self._store_surface(current_surface, current_surface_number)

    def _store_surface(self, surface, surface_number):
        """
        Helper method to store a surface in the dictionary with appropriate keys.
        """
        # Store by surface number
        self.surfaces[surface_number] = surface

        # Store by COMM name if available
        comm = getattr(surface, "COMM", None)
        if comm:
            self.surfaces[comm] = surface

        # Include STOP marker in sublist keys if applicable
        surface_name = [surface_number, comm] if comm else [surface_number]
        if getattr(surface, "STOP", False):
            surface_name.append("STOP")
            self.surfaces["STOP"] = surface  # Add STOP key for direct access
        surface_name = [elem for elem in surface_name if elem]  # Remove None
        self.surfaces[tuple(surface_name)] = surface

    def identify_stop_surface(self):
        """
        Identify the STOP surface by checking all surfaces for the STOP attribute.
        """
        for surface in self.surfaces.values():
            if isinstance(surface, Surface) and getattr(surface, "STOP", False):
                self.STOP = surface
                self.surfaces["STOP"] = surface  # Ensure STOP key is available
                break

    def parse_system_details(self):
        """
        Parses the file content to extract system details like ENPD, UNIT, WAVM, and FEFD.
        """
        if not self.file_content:
            raise ValueError("File content is empty. Did you call read_file()?")

        for line in self.file_content:
            line = line.strip()

            if line.startswith("ENPD"):  # Entrance pupil diameter
                try:
                    self.system_details.ENPD = float(line.split(" ", 1)[-1])
                except ValueError:
                    self.system_details.ENPD = None
            elif line.startswith("UNIT"):  # Units
                self.system_details.UNIT = line.split(" ", 1)[-1].split()
            elif line.startswith("WAVM"):  # Wavelengths
                self.system_details.WAVM.append(line.split(" ", 1)[-1])
            elif line.startswith("FEFD"):  # Fields
                self.system_details.FEF.append(line.split(" ", 1)[-1])

    def surface(self, key):
        """
        Retrieve a Surface object by its surface number, COMM name, or tuple key.
        """
        return self.surfaces.get(key)

    def get_surface_names(self):
        """
        Returns a list of sublists where each sublist contains the surface number,
        COMM name, and STOP if applicable.
        """
        result = []
        seen = set()

        for key, surface in self.surfaces.items():
            if isinstance(key, tuple):  # Surface names as tuples
                if key not in seen:
                    result.append(list(key))
                    seen.add(key)
            elif key.isdigit():  # Surface number only
                comm = getattr(surface, "COMM", None)
                surface_name = [key, comm] if comm else [key]
                if getattr(surface, "STOP", False):
                    surface_name.append("STOP")
                surface_name = [elem for elem in surface_name if elem]  # Remove None
                result.append(surface_name)
                seen.add(tuple(surface_name))

        return result
    

###################
###################
###################
 
def main():
    import numpy as np

    # Instantiate
    telescope = ZemaxFileParser('STP_TMA_Mark_12F+M1_Bending_SBTest_5_HC01.zmx')

    print('\nNAMES', telescope.get_surface_names())

    # Example usage
    SURF = telescope.surface('M1')
    
    # Access system details as attributes
    system_details = telescope.system_details

    print('\n')
    print('SURF:  ', SURF.PARM2)
    print('THIC:   ', SURF.DISZ)
    print('ENPD:    ', system_details.ENPD)
    print('STOP:', telescope.STOP)
    print('UNIT:', system_details.UNIT)

if __name__=='__main__':
    main()