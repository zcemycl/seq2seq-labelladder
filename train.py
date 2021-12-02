import argparse
import random
from models.seq2seqEncoderDecoder import *
from data.preprocess import *
from torch.utils.tensorboard import SummaryWriter
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_forcing_ratio = 0.5
MAX_LENGTH=100

class TrainingProcedure:
    def __init__(self,config):
        self.config = config
        self.max_length = MAX_LENGTH
        self._prepareData()
        self._initializeModels()
        self._initializeLogFolder()
        self._initializeModelFolder()
    def _prepareData(self):
        if not os.path.isfile(self.config.pairs_txt):
            print('[INFO] {} is not found. Proceed preprocessing csv file to txt.'.format(self.config.pairs_txt))
            process = PreprocessCSV(args)
            process.process2save()
        print('[INFO] {} is found. No preprocessing of csv required. Loading data...'.format(self.config.pairs_txt))
        input_vocab,output_vocab,pairs=getInputOutputVocabs(self.config.pairs_txt)
        self.input_vocab=input_vocab
        self.output_vocab=output_vocab
        self.pairs = pairs
        random.shuffle(self.pairs)
        self.trainpairs = self.pairs[:int(self.config.split*len(self.pairs))]
        self.testpairs = self.pairs[int(self.config.split*len(self.pairs)):]

        print('[INFO] Done.')
    def _initializeModels(self):
        print('[INFO] Loading models and optimizers...')
        self.encoder = EncoderRNN(self.input_vocab.n_words,hidden_size).to(device)
        self.attn_decoder = AttnDecoderRNN(hidden_size,self.output_vocab.n_words,dropout_p=0.1).to(device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.config.lr)
        self.decoder_optimizer = optim.SGD(self.attn_decoder.parameters(), lr=self.config.lr)
        self.criterion = nn.NLLLoss()
        print('[INFO] Done.')
    def _initializeLogFolder(self):
        print('[INFO] Initialize log writer...')
        pathsplit = self.config.logdir.split(os.sep)
        for i in range(1,len(pathsplit)+1):
            if not os.path.isdir(os.sep.join(pathsplit[:i])):
                os.mkdir(os.sep.join(pathsplit[:i]))
        self.writer = SummaryWriter(log_dir=self.config.logdir)
        print('[INFO] Done.')
    def _initializeModelFolder(self):
        print('[INFO] Initialize checkpoint folder...')
        pathsplit = self.config.checkpoint.split(os.sep)
        for i in range(1,len(pathsplit)+1):
            if not os.path.isdir(os.sep.join(pathsplit[:i])):
                os.mkdir(os.sep.join(pathsplit[:i]))
        print('[INFO] Done.')
    def _saveModel(self,epoch):
        torch.save(self.encoder,self.config.checkpoint+'/encoder-{}.pth'.format(epoch))
        torch.save(self.attn_decoder,self.config.checkpoint+'/attn_decoder-{}.pth'.format(epoch))

    def train(self,input_tensor,target_tensor):
        encoder_hidden = self.encoder.initHidden()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / target_length

    def evaluate(self,pair,max_length=MAX_LENGTH):
        loss = 0
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.input_vocab,pair[0], " ")
            target_tensor = tensorFromSentence(self.output_vocab, pair[1],',')

            input_length = input_tensor.size()[0]
            target_length = target_tensor.size(0)
            encoder_hidden = self.encoder.initHidden()
            encoder_outputs = torch.zeros(max_length,self.encoder.hidden_size,device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.attn_decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += self.criterion(decoder_output, target_tensor[di])
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_vocab.index2word[topi.item()])

            return decoded_words,loss

    def evalIters(self,epoch):
        cl = [0]*4 # L1 Acc, L2 Acc, L3 Acc, Total Acc
        test_loss_total = 0
        random.shuffle(self.testpairs)
        st = ""
        for i in tqdm(range(len(self.testpairs))):
            pair = self.testpairs[i]
            out_words,loss = self.evaluate(pair)
            test_loss_total+=loss
            tmp = pair[1].split(',')
            out_sent = ",".join(out_words[:-1])
            if i < 3:
                st+=("[Input] "+pair[0]+" |[Target] "+pair[1]+" |[Output] "+out_sent+"\n")
            if out_sent == pair[1]:
                cl[3]+=1
            for j in range(len(tmp)):
                if tmp[j]==out_words[j]:
                    cl[j]+=1
        self.writer.add_text('Text Example',st,epoch)
        return test_loss_total,cl

    def trainIters(self):
        train_loss_total = 0  # Reset every save interval

        training_pairs = [tensorsFromPair(random.choice(self.trainpairs),
                          self.input_vocab,self.output_vocab)
                          for i in range(self.config.iters)]
        print('[INFO] Start Training...')
        for i in tqdm(range(self.config.iters + 1)):
            training_pair = training_pairs[i - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor)
            train_loss_total += loss

            if i%self.config.saveInterval == 0:
                train_loss_avg = train_loss_total / self.config.saveInterval
                train_loss_total = 0
                test_loss_total,cl = self.evalIters(i)
                testSize = len(self.testpairs)
                print('[INFO] Saving losses, accuracies and test cases to Tensorboard...')
                self.writer.add_scalars('Loss',{'train_avg_loss':train_loss_avg,
                    'test_avg_loss':test_loss_total/testSize},i)
                self.writer.add_scalars('Test Accuracy',{'l1':cl[0]/testSize,
                    'l2':cl[1]/testSize,'l3':cl[2]/testSize,
                    'overall':cl[3]/testSize},i)
                print('[INFO] Done.')
                print('[INFO] Saving Model Weights...')
                self._saveModel(i)
                print('[INFO] Done.')

        print('[INFO] Done.')

def main(config):
    process = TrainingProcedure(config)
    process.trainIters()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--csv',type=str,default="/media/yui/Disk/data/util_task/classification_dataset.csv")
    p.add_argument('--pairs_txt',type=str,default="/media/yui/Disk/data/util_task/pairs.txt")
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--split',type=float,default=0.99)
    p.add_argument('--logdir',type=str,default="log/run")
    p.add_argument('--checkpoint',type=str,default="weight/run")
    p.add_argument('--iters',type=int,default=75000)
    p.add_argument('--saveInterval',type=int,default=1000)
    p.add_argument('--lr',type=float,default=0.01)
    args = p.parse_args()
    random.seed(args.seed)
    main(args)
