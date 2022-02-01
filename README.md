# pyneurone_tools

[![Python package](https://github.com/jkschluesener/pyneurone_tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/jkschluesener/pyneurone_tools/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/jkschluesener/pyneurone_tools/actions/workflows/python-publish.yml/badge.svg)](https://github.com/jkschluesener/pyneurone_tools/actions/workflows/python-publish.yml)

A simple python library to read Bittium NeurOne Tesla EEG system binary data

## Function overview

### Basic Usage

```python
from pyneurone_tools import neurone_tools
from pathlib import Path

session_path = Path('./exciting_project/exciting_data/2022-01-03T090018')
recording_id = 1

nt = neurone_tools(session_path, recording_id, channels='all')
```

### loading data

```python
data, channel_names = nt.load_data()
```

The data has the shape (samples, channels), unit is nanovolt.

### loading events

```python
events = nt.load_events()
```

Evens are returned as a pandas DataFrame.  
The columns are:

`revision`: Event marker revision number  
`type`: Type of event marker  
`source_port`: Source port number  
`channel_number`: Channel used for event marking  
`8bit_trigger_code`: The 8-bit TTL trigger code if the marker has one, otherwise None  
`start_sample_index`: sample point where trigger occured  
`stop_sample_index`: sample point where trigger ended  
`description_length`: length of event comment (string in recording system)  
`description_offset`: offset of event comment (string in recording system)  
`data_length`: length of additional data of event (any binary data)  
`data_offset`: offset of additional data of event (any binary data)  
`source_port_name`: name of the channel from channel_number  
`description`: name of the event marker type  

### sampling rate

```python
fsample = nt.get_sampling_rate()
```

### loading channel names

```python
channel_names = nt.channels_avail
```

### loading alternating current (AC) mode flags

```python
ac_mode = nt.get_ac_mode()
```

### loading filter parameters

```python
filter_settings = nt.get_filter_settings()
```

## Terminology

The following terms are used in this package

### Session

A session is the top-level term, think of it as the folder containing all the other files.  
It is started when you click 'Start' in the recording software and can be made up from several recordings.

### Recording

A recording is what you get when you press the 'Record' button and ends when you click the 'Stop Recording' button.  
Recording ID numbers are 1-indexed, as defined by the recording software.

### Binary File

A Recording can be split into several Binary files, as one can be specified in the NeurOne recording software.
Binary file IDs are 1-indexed, as this is given by the recording software.  
It is unikely that the user would ever need to access these files directly.

## Installation

### Supported Python Versions

This code is tested on python versions 3.7, 3.8, 3.9, and 3.10.  
Earlier versions could be usable, but need an older version of pandas.

### Using pip

```console
pip install neurone_tools
```

### pip directly from github

```bash
pip install git+https://github.com/jkschluesener/pyneurone_tools.git@master
```

### pip without PyPi

```console
cd <your_preferred_code_folder>
git clone https://github.com/jkschluesener/pyneurone_tools.git
pip install ./pyneurone_tools/
```

### Development mode

Link this repo's folder into your current python with the `-e` (external) flag.

```console
cd <your_preferred_code_folder>
git clone https://github.com/jkschluesener/pyneurone_tools.git
pip install -e ./pyneurone_tools/
```

or use ssh if you prefer

```console
cd <your_preferred_code_folder>
git clone git@github.com:jkschluesener/pyneurone_tools.git
pip install -e pyneurone_tools
```

Changes to this directory will be available without reinstall.  
Upon local changes of the code, you may have to re-import this package or maybe restart your python kernel.

## Adding functionality

Is there a funcitonality you would like to see in this package?  
Just open an Issue for us / me to discuss and implement it.

## License

This code is GPLv3 licensed.  

Layout of the binary files was taken from the official NeurOne manual.  
This code is also in parts derived from the original Bittium / Mega Electronics Ltd matlab toolbox ([official](https://www.bittium.com/medical/bittium-neurone) / [mirror](https://github.com/jkschluesener/neurone_tools_matlab)) to map integer ID codes to their string counterparts.
