import numpy as np
import json

import triton_python_backend_utils as pb_utils

from punctuate.punctuate_text import Punctuation

class TritonPythonModel:

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_TEXT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.all_punct = { 
            'hi': Punctuation('hi'),
            # 'gu': Punctuation('gu'),
            'mr': Punctuation('mr'),
            # 'pa': Punctuation('pa'),
            'bn': Punctuation('bn'),
            'or': Punctuation('or'),            
            'ta': Punctuation('ta'),
            'te': Punctuation('te'),
            # 'kn': Punctuation('kn'),
            'ml': Punctuation('en'),
            # 'ml': Punctuation('ml')
        }

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            # B X T
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text = [s[0].decode() for s in in_0.as_numpy()]
            
            # B X 1
            in_1 = pb_utils.get_input_tensor_by_name(request, "LANG_ID")
            # print([s.decode("utf-8") for s in in_2.as_numpy()])
            # NOTE: Assumption is that the "lang" in a request-batch remains same
            lang = in_1.as_numpy()[0].decode("utf-8")

            sent = np.array(self.all_punct[lang].punctuate_text(input_text))
            
            out_tensor_0 = pb_utils.Tensor("OUTPUT_TEXT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
