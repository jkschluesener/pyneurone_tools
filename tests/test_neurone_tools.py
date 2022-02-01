import pytest
from unittest import TestCase
from _pytest.monkeypatch import MonkeyPatch
from ..neurone_tools import neurone_tools
import numpy as np
import pandas as pd
from pathlib import Path

# Not all functions can be tested straight away, as most require some file structure. So we monkeypatch them.
# However we want to make sure that assertions are raised when needed, so we can't get around unittest.TestCase.
# unittest.TestCase methods cannot directly receive fixture function arguments.
# Thus the more complicated setup to get monkeypatch going with importing it.
# Still, this won't solve the issue with testing loading of files - either from binary or xml.
# These tests don't cover these cases.


class test_neurone_tools(TestCase):

    def setUp(self):
        self.monkeypatch = MonkeyPatch()

    def test_channel_requests(self):
        '''test various channel requests, if they exist or not.'''

        def mock_channels_avail(*args, **kwargs):
            '''Monkeypatching mock function to return a numpy array of known channel names'''
            return np.array(['Input 1', 'Input 2', 'Input 3'])

        def mock_path_exists(*args, **kwargs):
            '''Monkeypatching mock function to return `True` if pathlib.Path instance is asking if a path exists'''
            return True

        def test_requests(channels):
            nt = neurone_tools('test_path', 50, channels='all')
            nt.channels_request = channels
            return nt.check_channels_exist()

        # apply monkeypatches
        self.monkeypatch.setattr(neurone_tools, 'load_channel_names', mock_channels_avail)
        self.monkeypatch.setattr(Path, 'exists', mock_path_exists)

        # prepare test arrays
        channels_all = mock_channels_avail()
        channels_one = mock_channels_avail()[0]
        channels_few = mock_channels_avail()[0:2]
        channels_toomany = np.append(mock_channels_avail(), 'Input 99')

        self.assertTrue(test_requests(channels_all))
        self.assertTrue(test_requests(channels_one))
        self.assertTrue(test_requests(channels_few))
        self.assertFalse(test_requests(channels_toomany))

    def test_get_channel_idx(self):
        '''test if the correct channel indizes are returned for a query'''

        def mock_channels_avail(*args, **kwargs):
            '''Monkeypatching mock function to return a numpy array of known channel names'''
            return np.array(['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'])

        def mock_path_exists(*args, **kwargs):
            '''Monkeypatching mock function to return `True` if pathlib.Path instance is asking if a path exists'''
            return True

        # apply monkeypatches
        self.monkeypatch.setattr(neurone_tools, 'load_channel_names', mock_channels_avail)
        self.monkeypatch.setattr(Path, 'exists', mock_path_exists)

        # prepare test arrays
        channels_exist = mock_channels_avail()

        channels_sample = np.array([0, 2, 3])
        channels_search = channels_exist[channels_sample]

        nt = neurone_tools('test_path', 50, channels='all')
        channels_idx = nt.get_channel_idx(channel_search=channels_search, channel_lookup=channels_exist)

        self.assertTrue(np.array_equal(channels_idx, channels_sample))

    def test_sort_data_by_channels(self):
        '''test if data is sorted correctly by channel name'''

        def mock_channels_avail(*args, **kwargs):
            '''Monkeypatching mock function to return a numpy array of known channel names'''
            return np.array(['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'])

        def mock_path_exists(*args, **kwargs):
            '''Monkeypatching mock function to return `True` if pathlib.Path instance is asking if a path exists'''
            return True

        # apply monkeypatches
        self.monkeypatch.setattr(neurone_tools, 'load_channel_names', mock_channels_avail)
        self.monkeypatch.setattr(Path, 'exists', mock_path_exists)

        # prepare test arrays
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array(['Input 3', 'Input 2', 'Input 1'])

        data_expect = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
        labels_expect = np.array(['Input 1', 'Input 2', 'Input 3'])

        nt = neurone_tools('test_path', 50, channels='all')
        data_sort, labels_sort = nt.sort_data_by_channels(data, labels)

        self.assertTrue(np.array_equal(data_expect, data_sort))
        self.assertTrue(np.array_equal(labels_expect, labels_sort))

    def test_generate_path(self):
        '''test typical generate_path usecases'''

        paths_test = ['DataSetProtocol', 'DataSetSession', 'Protocol', 'Session']

        test_path = Path('test_path')
        recording_id = 50
        binary_id = 1

        nt = neurone_tools(test_path, recording_id, channels='all')

        for ipath in paths_test:
            path_returned = nt.generate_path(ipath, recording_id)
            path_correct = test_path / Path(ipath + '.xml')
            self.assertEqual(path_correct, path_returned)

        path_correct = test_path / f'{recording_id}/eventData.bin'
        path_returned = nt.generate_path('eventData')
        self.assertEqual(path_correct, path_returned)

        path_correct = test_path / f'{recording_id}/{binary_id}.bin'
        path_returned = nt.generate_path('Data', binary_id)
        self.assertEqual(path_correct, path_returned)

    def test_events_annotation(self):
        '''test events parsing'''

        def mock_channels_avail(*args, **kwargs):
            '''Monkeypatching mock function to return a numpy array of known channel names'''
            return np.array(['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'])

        def mock_path_exists(*args, **kwargs):
            '''Monkeypatching mock function to return `True` if pathlib.Path instance is asking if a path exists'''
            return True

        def mock_events_binary(*args, **kwargs):
            '''Monkeypatching mock function to return a known events DataFrame'''

            df = pd.DataFrame([[5, 0, 4, 3, 0, 2, 147771, 147771, 0, 0, 0, 0, -2086615908, 1101235444, 0, 0],
                               [5, 0, 4, 3, 0, 4, 147906, 147906, 0, 0, 0, 0, -543112036, 1101235654, 0, 0],
                               [5, 0, 4, 3, 0, 8, 148019, 148019, 0, 0, 0, 0, -1851734884, 1101235831, 0, 0],
                               [5, 0, 4, 3, 0, 16, 153819, 153819, 0, 0, 0, 0, 799065244, 1101244894, 0, 0],
                               [5, 0, 4, 3, 0, 32, 153883, 153883, 0, 0, 0, 0, -1683962724, 1101244994, 0, 0]],
                              columns=[
                                  'revision', 'RFU0', 'type', 'source_port', 'channel_number', '8bit_trigger_code',
                                  'start_sample_index', 'stop_sample_index', 'description_length', 'description_offset',
                                  'data_length', 'data_offset', 'RFU1', 'RFU2', 'RFU3', 'RFU4'])
            return df

        # apply monkeypatches
        self.monkeypatch.setattr(neurone_tools, 'load_channel_names', mock_channels_avail)
        self.monkeypatch.setattr(neurone_tools, 'load_binary_events', mock_events_binary)
        self.monkeypatch.setattr(Path, 'exists', mock_path_exists)

        df_expect = pd.DataFrame([[5, 4, 3, 0, 2, 147770, 147771, 0, 0, 0, 0, '8bit', None, '8bit_2'],
                                  [5, 4, 3, 0, 4, 147905, 147906, 0, 0, 0, 0, '8bit', None, '8bit_4'],
                                  [5, 4, 3, 0, 8, 148018, 148019, 0, 0, 0, 0, '8bit', None, '8bit_8'],
                                  [5, 4, 3, 0, 16, 153818, 153819, 0, 0, 0, 0, '8bit', None, '8bit_16'],
                                  [5, 4, 3, 0, 32, 153882, 153883, 0, 0, 0, 0, '8bit', None, '8bit_32']],
                                 columns=[
                                     'revision', 'type', 'source_port', 'channel_number', '8bit_trigger_code',
                                     'start_sample_index', 'stop_sample_index', 'description_length', 'description_offset',
                                     'data_length', 'data_offset', 'source_port_name', 'comment', 'description'])

        nt = neurone_tools('test_path', 50, channels='all')
        df_result = nt.load_events()

        self.assertTrue(df_result.equals(df_expect))
