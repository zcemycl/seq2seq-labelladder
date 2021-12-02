import argparse
import random
from train import *
from models.seq2seqEncoderDecoder import *
from data.preprocess import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH=100

def initialize(args):
    print('[INFO] Initializing models, sentence and dictionaries...')
    input_vocab,output_vocab,_=getInputOutputVocabs(args.pairs_txt)
    encoder = torch.load(args.weightE).to(device)
    decoder = torch.load(args.weightD).to(device)
    encoder.eval()
    decoder.eval()
    filtered_sent = PreprocessCSV.filterMethod(args.sentence,input_vocab.word2index)
    print('[INFO] Done.')
    return input_vocab,output_vocab,encoder,decoder,filtered_sent


def predict(args,input_vocab,output_vocab,encoder,decoder,filtered_sent):
    print('[INFO] Predicting {}...'.format(args.sentence))
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_vocab,filtered_sent, " ")
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(MAX_LENGTH,encoder.hidden_size,device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(4):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().detach()
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_vocab.index2word[topi.item()])
    return decoded_words[:-1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weightE',type=str,default='weight/run/encoder-75000.pth')
    p.add_argument('--weightD',type=str,default='weight/run/attn_decoder-75000.pth')
    p.add_argument('--pairs_txt',type=str,default='/media/yui/Disk/data/util_task/pairs.txt')
    p.add_argument('--sentence',type=str,default="Giovanni Giacomo Semenza (18 July 1580 – 1638) was an Italian painter of the early Baroque period. Born in Bologna and also known as Giacomo Sementi. He was a pupil of the painter Denis Calvaert, then of Guido Reni. Among his pupils were Giacinto Brandi. He painted a Christ the Redeemer for the church of St. Catherine in Bologna.") # Agent Artist Painter
    args = p.parse_args()

    input_vocab,output_vocab,encoder,decoder,filtered_sent = initialize(args)
    decoded_words = predict(args,input_vocab,output_vocab,encoder,decoder,filtered_sent)
    print("[INFO] Result: ",decoded_words)
    print("[INFO] Done.")


