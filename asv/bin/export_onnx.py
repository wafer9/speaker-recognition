import onnxruntime
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import soundfile, numpy, random
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import argparse, glob, os, torch, warnings, time, yaml
from asv.nnet.ecapa import ECAPA_TDNN, AAMsoftmax, SVModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model', required=True, help='save model dir')
    args = parser.parse_args()

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    ecapatdnn = ECAPA_TDNN(configs['linear_units'])
    ecapaloss = AAMsoftmax(configs['output_dim'])
    model = SVModel(ecapa_tdnn=ecapatdnn, aam_softmax=ecapaloss)

    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    ecapatdnn.eval()
    audio, sr = soundfile.read('/data/joe/data/voxceleb1/id10001/1zcIwhmdeo4/00001.wav')
    speech = torch.FloatTensor(audio).unsqueeze(0)
    mat = kaldi.mfcc(
        speech,
        num_ceps=40,
        num_mel_bins=40,
        frame_length=25,
        frame_shift=10,
        sample_frequency=sr)


    speech = mat.unsqueeze(0)
    with torch.no_grad():
        y = ecapatdnn(speech)
    onnx_path = args.model + ".onnx"
    torch.onnx.export(ecapatdnn,
                      (speech),
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['speech'],
                      output_names=['output'],
                      dynamic_axes={'speech': {1: 'T'}},
                      verbose=True,
                      )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(speech)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(to_numpy(y))
    print(ort_outs[0])
    np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported encoder model has been tested with ONNXRuntime, and the result looks good!")

    model_fp32 = onnx_path
    model_quant = args.model+ ".quant.onnx"
    quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

