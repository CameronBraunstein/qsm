from utils.quantum.binary_encoding_qubo_codec import BinaryEncodingQUBOCodec
from utils.quantum.one_hot_qubo_codec import OneHotQUBOCodec

def get_qubo_codec(args):
    if args.qubo_encoding_type == 'binary':
        qc = BinaryEncodingQUBOCodec(args)
    elif args.qubo_encoding_type == 'one_hot':
        qc = OneHotQUBOCodec(args)
    else:
        raise Exception('No other type of converter supported')
    return qc