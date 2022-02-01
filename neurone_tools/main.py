# ------------------------------------------------------------------------
# Read Bittium (formerly Mega Electronics) NeurOne binary data into python
#
# A part of two functions is derived from Bittium's Matlab toolbox (GPLv3) and are marked as such.
#
# (c) 2022 Jan K. Schluesener
# Released under GNU Public License (GPL)
#
# github.com/jkschluesener
#
# ------------------------------------------------------------------------


import numpy as np
from typing import Union, Tuple, List, Dict, OrderedDict
from numpy.typing import ArrayLike
from os import PathLike
from pathlib import Path
import xmltodict
import pandas as pd
from natsort import index_natsorted


class neurone_tools:

    def __init__(self, session_path: Union[str, PathLike], recording_id: int, channels='all'):
        """Initialize pyneurone to a session and recording from which you wish to load data.
        Some assertions check the inputs.

        Parameters
        ----------
        session_path : Union[str,PathLike]
            Path to the session you wish to load recordings from
        recording_id : int
            1-indexed ID number of the recording to load
        channels : str, optional
            string of one channel / list / array of channels you want to load. By default 'all'.
        """

        session_path = Path(session_path)

        self.session_path = session_path
        self.channels_avail = self.load_channel_names()
        self.recording_id = recording_id

        channels = np.asarray(channels)

        if np.array_equal(channels, np.array('all')):
            self.channels_request = self.channels_avail
        else:
            self.channels_request = np.atleast_1d(np.asarray(channels))

        assert self.session_path.exists(), 'The requested `session_path` does not exist'
        assert isinstance(self.recording_id, int), '`recording_id` must be an integer'
        assert all([isinstance(c, str) for c in self.channels_request]), '`channels` must be str or list of str or ArrayLike of str'
        assert self.check_channels_exist(), 'Not all requested `channels` exist in the session'

    ##################################################
    # Channel-Related Functions
    ##################################################

    def load_channel_names(self) -> ArrayLike:
        """Returns the channel names of a session.

        Returns
        -------
        ArrayLike
            Array of channel names
        """

        protocol_path = self.generate_path('DataSetProtocol')
        protocol = self.read_xml(protocol_path)
        
        df = pd.DataFrame(columns=['input_name', 'input_position'])
        df['input_name'] = np.atleast_1d(np.array([str(i['Name']) for i in protocol['DataSetProtocol']['TableProtocolInput']]))
        df['input_position'] = np.atleast_1d(np.array([int(i['InputNumber']) for i in protocol['DataSetProtocol']['TableProtocolInput']]))
        
        channel_names = df.sort_values('input_position')['input_name'].values

        return channel_names

    def check_channels_exist(self) -> bool:
        """Check if all channels in self.channels_request are available in a session, self.channels_avail

        Returns
        -------
        bool
            True: All channels are available. False: At least one channel does not exist.
        """

        if np.array_equal(self.channels_request, self.channels_avail):
            return True

        return np.all(np.in1d(self.channels_request, self.channels_avail))

    def get_channel_idx(self, channel_search: ArrayLike, channel_lookup: ArrayLike) -> ArrayLike:
        """Return the indizes of channel_search in channel_lookup. The indizes are sorted by channel_search.

        Parameters
        ----------
        channel_search : ArrayLike
            Channels whose ID is needed of shape (n,)
        channel_lookup : ArrayLike
            Channel List of shape (m,)

        Returns
        -------
        ArrayLike
            Indices of channel_search in channel_lookup, sorted by channel_search
        """

        assert np.unique(channel_search).shape[0] == np.unique(channel_search).shape[0], 'There are duplicates in channel_search'
        assert np.unique(channel_lookup).shape[0] == np.unique(channel_lookup).shape[0], 'There are duplicates in channel_lookup'
        assert np.all(np.isin(channel_search, channel_lookup)), 'Not all requested channel_search exist in channel_lookup'

        return np.where(np.isin(channel_lookup, channel_search))[0][np.argsort(channel_search)]

    def sort_data_by_channels(self, data: ArrayLike, channels: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Sorts a data array and channel name array to be both sorted ascending.

        Parameters
        ----------
        data : ArrayLike
            Data to be sorted
        channels : ArrayLike
            Channel names to be sorted

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            Sorted Data and Channel Names
        """

        channel_index = index_natsorted(channels)

        channels = channels[channel_index]
        data = data[:, channel_index]

        return data, channels

    ##################################################
    # Binary Data File Readers
    ##################################################

    def load_data(self) -> Tuple[ArrayLike, ArrayLike]:
        """Loads a recording. Should a recording be comprised of several binary files, they are concatenated.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            data(samples,channels) and Channel Names. Unit is nanovolt.
        """

        session_data_path = self.generate_path('DataSetSession')
        session_data = self.read_xml(session_data_path)

        meta = session_data['DataSetSession']['TableSessionPhase']

        if isinstance(meta, OrderedDict):
            n_binary_files = 1
        else:
            n_binary_files = int(meta[self.recording_id-1]['FileCount'])

        data_out = []
        for ibinary in range(1, n_binary_files + 1):
            data, channels = self.load_binary_data(ibinary)
            data_out.append(data)

        data_out = np.concatenate(data_out)

        return data_out, channels
    
    def load_binary_data(self, binary_id: int) -> Tuple[ArrayLike, ArrayLike]:
        """Loads a binary data file from disk.

        Parameters
        ----------
        binary_id : int
            ID number of the binary file that should be loaded.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            data(samples,channels) and channel names. Unit is nanovolt.
        """

        fpath_load = self.generate_path('data', binary_id=binary_id)

        if np.array_equal(self.channels_request, self.channels_avail):
            data = np.fromfile(fpath_load, dtype=np.int32)
            data = data.reshape((-1, len(self.channels_avail)))

        else:
            chan_idx = self.get_channel_idx(self.channels_request, self.channels_avail)

            data = np.memmap(fpath_load, dtype=np.int32)
            data = data.reshape(-1, len(self.channels_avail))

            data = data[:, chan_idx]

        data, channels = self.sort_data_by_channels(data, self.channels_request)

        data = data.astype('float')

        # microvolt -> nanovolt
        data /= 1000.

        data = self.calibrate_data(data)

        return data, channels

    def calibrate_data(self, data: ArrayLike) -> ArrayLike:
        """Transfers data from the raw data range to the calibrated range.
        This function mutates the input array.

        Parameters
        ----------
        data : ArrayLike
            data(samples,channels) in raw data range.

        Returns
        -------
        ArrayLike
            data(samples,channels) in raw data range.
        """

        fpath = self.generate_path('DataSetProtocol')
        session_data = self.read_xml(fpath)

        channel_names = np.atleast_1d(np.array([str(i['Name']) for i in session_data['DataSetProtocol']['TableProtocolInput']]))
        idx_use = np.array([np.where(i==channel_names)[0][0] for i in self.channels_request])

        calib_min = np.atleast_1d(np.array([float(i['RangeAsCalibratedMinimum']) for i in session_data['DataSetProtocol']['TableProtocolInput']]))[idx_use]
        calib_max = np.atleast_1d(np.array([float(i['RangeAsCalibratedMaximum']) for i in session_data['DataSetProtocol']['TableProtocolInput']]))[idx_use]
        range_min = np.atleast_1d(np.array([float(i['RangeMinimum']) for i in session_data['DataSetProtocol']['TableProtocolInput']]))[idx_use]
        range_max = np.atleast_1d(np.array([float(i['RangeMaximum']) for i in session_data['DataSetProtocol']['TableProtocolInput']]))[idx_use]

        data -= range_min
        data += calib_min
        data /= range_max - range_min
        data *= calib_max - calib_min

        return data

    ##################################################
    # Handling Paths, Parsing Plain Text
    ##################################################

    def read_xml(self, fpath: Union[PathLike, str]) -> OrderedDict:
        """Read a xml file from path as OrderedDicts.

        Parameters
        ----------
        fpath : Union[PathLike,str]
            Path to the xml file

        Returns
        -------
        OrderedDict
            xml file as OrderedDicts
        """

        with open(fpath, 'r') as file:
            data = file.read()

        protocol = xmltodict.parse(data)

        return protocol

    def generate_path(self, destination: str, binary_id: int = 1) -> PathLike:
        """Generate the path to a specified file of a recording session.
        More files are coded here than are needed for the current scope of this package.

        Parameters
        ----------
        destination : str
            File request
        binary_id : int
            1-indexed id of the binary file that will be loaded as is noted in the session metadata, by default 1.
            Not used for anything but data loading.

        Returns
        -------
        PathLike
            Path to the requested datafile.

        Raises
        ------
        ValueError
            Unknown file requested
        """

        assert isinstance(destination, str), '`destination` to a file needs to be str'
        assert isinstance(binary_id, int), '`binary_id` needs to be int'

        if destination in ['DataSetProtocol', 'datasetprotocol']:
            out = self.session_path / 'DataSetProtocol.xml'
        elif destination in ['DataSetSession', 'datasetsession']:
            out = self.session_path / 'DataSetSession.xml'
        elif destination in ['HeadBoxVoltages', 'headboxvoltages', 'headboxvolt']:
            out = self.session_path / 'HeadBoxVoltages.txt'
        elif destination in ['ImpedanceData', 'impedancedata', 'impedance']:
            out = self.session_path / 'ImpedanceData.txt'
        elif destination in ['Protocol', 'protocol']:
            out = self.session_path / 'Protocol.xml'
        elif destination in ['Session', 'session']:
            out = self.session_path / 'Session.xml'
        elif destination in ['eventData', 'eventdata']:
            out = self.session_path / f'{self.recording_id}/eventData.bin'
        elif destination in ['eventDescriptions', 'eventdescriptions', 'eventdesc']:
            out = self.session_path / f'{self.recording_id}/eventDescriptions.bin'
        elif destination in ['events']:
            out = self.session_path / f'{self.recording_id}/events.bin'
        elif destination in ['Data', 'data']:
            out = self.session_path / f'{self.recording_id}/{binary_id}.bin'
        else:
            raise ValueError('Unknown file requested: %s', destination)

        assert out.exists(), 'Requested file does not exist!'

        return out

    ##################################################
    # Reading Events
    ##################################################

    def load_binary_events(self) -> pd.DataFrame:
        """Loads the binary event data of a recording.

        Returns
        -------
        pd.DataFrame
            The event data of a session, as a pandas DataFrame
        """

        # generate needed paths
        events_path = self.generate_path('events')

        # custom datatype used in this binary file
        dt = np.dtype([
            ('revision', np.int32),
            ('RFU0', np.int32),
            ('type', np.int32),
            ('source_port', np.int32),
            ('channel_number', np.int32),
            ('8bit_trigger_code', np.int32),
            ('start_sample_index', np.uint64),
            ('stop_sample_index', np.uint64),
            ('description_length', np.uint64),
            ('description_offset', np.uint64),
            ('data_length', np.uint64),
            ('data_offset', np.uint64),
            ('RFU1', np.int32),
            ('RFU2', np.int32),
            ('RFU3', np.int32),
            ('RFU4', np.int32)])

        # read the binary file with the custom datatype and cast as a dataframe
        events = np.fromfile(events_path, dtype=dt)
        events = pd.DataFrame(events)
    
        return events

    def load_events(self) -> pd.DataFrame:
        """Returns the annotated event data of a recording.

        Returns
        -------
        pd.DataFrame
            Event data of a recording.
            Column names:
                revision: Event marker revision number
                type: Type of event marker
                source_port: Source port number
                channel_number: Channel used for event marking
                8bit_trigger_code: The 8-bit TTL trigger code if the marker has one, otherwise None
                start_sample_index: sample point where trigger occured
                stop_sample_index: sample point where trigger ended
                description_length: length of event comment (string in recording system)
                description_offset: offset of event comment (string in recording system)
                data_length: length of additional data of event (any binary data)
                data_offset: offset of additional data of event (any binary data)
                source_port_name: name of the channel from channel_number
                description: name of the event marker type
        """

        events = self.load_binary_events()

        # drop columns `Reserved for Future Updates` aka `RFU`
        events.drop(columns=list(events.filter(regex='RFU*')), inplace=True)

        # generate source port name from the source port number
        events['source_port_name'] = np.nan
        events['source_port_name'] = events.apply(self.get_source_port_name, axis=1)

        # extract the comments
        events['comment'] = np.nan
        events['comment'] = events.apply(self.get_event_comments, axis=1)

        # make `start_sample_index` is 0-indexed, and leave `stop_sample_index` 1-indexed
        # this way the python-style readout array[start:stop] will return the correct data.
        events['start_sample_index'] = events['start_sample_index'] - 1

        # get the event descriptions
        events['description'] = np.nan
        events['description'] = events.apply(self.get_event_descriptions, axis=1)

        return events

    def get_source_port_name(self, c: pd.Series) -> str:
        """Returns the name of a source port as a string. This function is meant to be applied to an events dataframe.

        The source port mapping of id number to name was taken from Bittium's matlab toolbox.

        Parameters
        ----------
        c : pd.Series
            Apply this function to an events dataframe

        Returns
        -------
        str
            Name of the source port
        """

        # Lookup of the source port names, as per Bittium's matlab toolbox
        sourcelut = {
            0: None,
            1: 'a',
            2: 'b',
            3: '8bit',
            4: 'syncbox_button',
            5: 'syncbox_external'}

        # Set the name if known, or parse the binary file to get the software component name.
        if c['source_port'] in sourcelut.keys():
            return sourcelut[c.source_port]
        elif c['source_port'] == 6:
            if c['description_length'] > 0:
                return self.get_software_component_name(c)
            else:
                return 'software'
        else:
            return 'unknown'

    def get_software_component_name(self, c: pd.Series) -> str:
        """Returns the name of a software component.

        Parameters
        ----------
        c : pd.Series
            Apply this function to an events dataframe

        Returns
        -------
        str
            Name of the software component
        """

        descriptions_path = self.generate_path('eventDescriptions')

        event_desc = np.fromfile(descriptions_path, dtype=np.int16)

        start = c['description_offset'] - 1
        length = c['description_length']

        # convert bytestring to string
        e = event_desc[start:start + length].decode("utf-8")

        return 'software_' + e

    def get_event_descriptions(self, c: pd.Series) -> str:
        """Returns event descriptions.

        The event description mapping of id number to name was taken from Bittium's matlab toolbox.

        Parameters
        ----------
        c : pd.Series
            Apply this function to an events dataframe

        Returns
        -------
        str
            Event description
        """

        # Lookup of the event descriptions, as per Bittium's matlab toolbox
        eventlut = {
            0: None,
            1: 'stimulation',
            2: 'video',
            3: 'mute',
            4: '8bit_' + str(c['8bit_trigger_code']),
            5: 'out',
            7: 'response_single_start',
            8: 'response_single_end',
            9: 'response_average_start',
            10: 'response_average_end',
            100: 'mep',
            2147483638: 'response_data_start',
            2147483639: 'response_data_end'}

        if c['type'] in eventlut.keys():
            return eventlut[c['type']]
        elif c['type'] == 6:
            return self.get_event_comments(c)
        else:
            return 'unknown'

    def get_event_comments(self, c: pd.Series) -> str:
        """Load the event comments from binary.

        Parameters
        ----------
        c : pd.Series
            Apply this function to an events dataframe

        Returns
        -------
        str
            Event comment
        """

        # If there is no event description, return None
        if c['data_length'] == 0:
            return None

        event_path = self.generate_path('eventData')

        event_desc = np.fromfile(event_path, dtype=np.int16)

        # Get annotation from this row
        start = c['data_offset'] - 1
        length = c['data_length']

        # convert bytestring to string
        comment = event_desc[start:start + length].decode("utf-8")

        return comment

    ##################################################
    # General Functions
    ##################################################

    def get_channel_info(self) -> List[Dict]:
        """Load the channel info of a session

        Returns
        -------
        List[Dict]
            Channel metadata
        """

        # generate path to dataset protocol file
        metadata_path = self.generate_path('DataSetProtocol')
        # read the protocol data
        metadata = self.read_xml(metadata_path)
        # get the channel metadata
        channel_info = metadata['DataSetProtocol']['TableProtocolInput']
        # convert list of OrderedDict to list of dicts
        channel_info = [dict(x) for x in channel_info]

        return channel_info

    def get_sampling_rate(self) -> float:
        """Get the sampling rate of a session

        Returns
        -------
        float
            Sampling rate
        """

        # generate path to dataset protocol file
        metadata_path = self.generate_path('DataSetProtocol')
        # read the protocol data
        metadata = self.read_xml(metadata_path)
        # get the sampling rate
        sampling_rate = float(metadata['DataSetProtocol']['TableProtocol']['ActualSamplingFrequency'])

        return sampling_rate
    
    def get_ac_mode(self) -> dict:
        """Returns the AC mode of the channels - True or False

        Returns
        -------
        dict
            dict with channels names as keys, with booleans indicating wether a channel was recorded in AC mode or not
        """

        # generate path to dataset protocol file
        fpath = self.generate_path('DataSetProtocol')
        # read the protocol data
        xml = self.read_xml(fpath)
        # ac mode
        ac = {i['Name']: bool(i['AlternatingCurrent']) for i in xml['DataSetProtocol']['TableProtocolInput']}
        
        ac = {k: ac[k] for k in self.channels_request}

        return ac
    
    def get_filter_settings(self) -> dict:
        """Returns the filtering applied to the data

        Returns
        -------
        dict
            dict with channels names as keys, with a string containing filter settings
        """

        # generate path to dataset protocol file
        fpath = self.generate_path('DataSetProtocol')
        # read the protocol data
        xml = self.read_xml(fpath)
        # ac mode
        flt = {i['Name']: i['Filter'] for i in xml['DataSetProtocol']['TableProtocolInput']}
        
        flt = {k: flt[k] for k in self.channels_request}

        return flt
