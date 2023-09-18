import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import time
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2 
from tqdm import tqdm
import tensorflow_hub as hub
from fuzzywuzzy import fuzz


# Loading model
use_model_path = "/Users/xaviertang/Desktop/GoogleUSE/universal-sentence-encoder-qa_3"
t = time.time()
module = hub.load(use_model_path)
print (round((time.time()-t), 3), "secs.")

# init encoders 
question_encoder = module.signatures['question_encoder']
response_encoder = module.signatures['response_encoder']
neg_response_encoder = module.signatures['response_encoder']

# show Google USE layers
print(module.variables)

# layers to be fine-tuned
v = ['QA/Final/Response_tuning/ResidualHidden_1/AdjustDepth/projection/kernel']
var_finetune = [x for x in module.variables for vv in v if vv in x.name]

# optimiser
adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-07)

def triplet_loss(anchor_vector, positive_vector, negative_vector, margin):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    :type anchor_vector: tf.Tensor
    :type positive_vector: tf.Tensor
    :type negative_vector: tf.Tensor
    :type metric: str
    :type margin: float
    :param anchor_vector: The anchor vector in this use case should be the encoded query. 
    :param positive_vector: The positive vector in this use case should be the encoded response. 
    :param negative_vector: The negative vector in this use case should be the wrong encoded response. 
    :param metric: Specify loss function
    :param margin: Margin parameter in loss function. See link above. 
    :return: the triplet loss value, as a tf.float32 scalar.
    """
    cosine_distance = tf.keras.losses.CosineSimilarity(axis=1)
    d_pos = cosine_distance(anchor_vector, positive_vector)
    d_neg = cosine_distance(anchor_vector, negative_vector)
    #print("d_pos: ", d_pos)
    #print("d_neg: ", d_neg)
    loss = tf.maximum(0., margin + d_pos - d_neg)
    #print("triplet_loss: ", loss)
    loss = tf.reduce_mean(loss)
    print("triplet_loss_reduce_mean: ", loss)
    return loss

def finetune_weights(question, 
                     answer,
                     neg_answer,
                     question_encoder,
                     response_encoder,
                     neg_response_encoder,
                     var_finetune,
                     optimizer,
                     margin=0.3,
                     loss='triplet'):
                         #context=[], 
                         #neg_answer=[],
                         #neg_answer_context=[], 
                         #label=[]):
        """
        Finetune the model with GradientTape

        :type question: list of str
        :type answer: list of str
        :type context: list of str
        :type neg_answer: list of str
        :type neg_answer_context: list of str
        :type margin: float
        :type label: list of int
        :type loss: str
        :param question: List of string queries
        :param answer: List of string responses
        :param context: List of string response contexts, this is applicable to the USE model
        :param neg_answer: List of string responses that do not match with the queries. This is applicable for triplet / contrastive loss.
        :param neg_answer_context: Similar to neg_answer for the USE model to ingest
        :param label: List of int
        :param margin: Marrgin tuning parameter for triplet / contrastive loss
        :param loss: Specify loss function
        :return:  numpy array of mean loss value
        """
        cost_history = []
        with tf.GradientTape() as tape:
            # tape.watch(var_finetune)
            # get encodings
            question_embeddings = question_encoder(
                tf.constant(question)
            )['outputs']

            response_embeddings = response_encoder(
                input=tf.constant(answer),
                context=tf.constant(answer)
            )['outputs']

            #print(question_embeddings)
            #print(response_embeddings)

            
            if loss == 'cosine':
                """
                # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity

                """
                cost = tf.keras.losses.CosineSimilarity(axis=1)
                cost_value = cost(question_embeddings, response_embeddings)
                
            elif loss == 'triplet':
                """
                Triplet loss uses a non-official self-implementated loss function outside of TF based on cosine distance

                """
                neg_response_embeddings = neg_response_encoder(
                    input=tf.constant(neg_answer),
                    context=tf.constant(neg_answer)
                )['outputs']

                cost_value = triplet_loss(
                    question_embeddings,
                    response_embeddings,
                    neg_response_embeddings,
                    margin=margin
                )

        # record loss
        cost_history.append(cost_value.numpy().mean())
        #print("cost_value: ", cost_value)
        #print(var_finetune)
        # apply gradient
        grads = tape.gradient(cost_value, var_finetune)
        #print(grads)
        #print(type(grads))
        optimizer.apply_gradients(zip(grads, var_finetune))

        return cost_value.numpy().mean()


# testing samples
questions_list = ["International Criminal Court", "Trade Facilitation"]
responses_list = ["I spoke before this body last year (see A/72/PV.3) and warned that the Human Rights Council had become a grave embarrassment to this institution, shielding egregious human rights abusers while bashing America and its many friends. Our Ambassador to the United Nations, Nikki Haley, laid out a clear agenda for reform, but despite reported and repeated warnings, no action at all was taken. The United States took the only responsible course. We withdrew from the Human Rights Council, and we will not return until real reform is enacted. For similar reasons, the United States will provide no support or recognition to the International Criminal Court (ICC). As far as America is concerned, the ICC has no jurisdiction, no legitimacy and no authority. The ICC claims near-universal jurisdiction over the citizens of every country, violating all principles of justice, fairness and due process. We will never surrender America’s sovereignty to an unelected, unaccountable global bureaucracy. America is governed by Americans. We reject the ideology of globalism and we embrace the doctrine of patriotism. Around the world, responsible nations must defend against threats to sovereignty not just from global governance, but also from other new forms of coercion and domination.",
                 "We are committed to strengthening trade facilitation which is a necessary pre-condition for sustained and deeper economic integration. We encourage all Member States to take the necessary steps to enable the ASEAN-wide implementation of Self-Certification Program by 2015. We acknowledge the progress made in the pilot implementation of the ASEAN Single Window, but we also recognize that for substantial progress to be made, Member States should implement their National Single Windows and rapidly put in place the needed legal and operational architecture to fully operationalize the ASW. We also urge progress in ratification and entry into force of various customs and transport protocols and agreements, particularly Protocol 2 (Designation of Frontier Posts) and Protocol 7 (Customs Transit System)."]
neg_response_list = ["Like those who met us before, our time is one of great contests, high stakes and clear choices. The essential divide that runs all around the world and throughout history is once again thrown in to stark relief. It is the divide between those whose thirst for control deludes them into thinking they are destined to rule over others and those people and nations who want only to rule themselves.",
                    "The most important difference in America’s new approach to trade concerns our relationship with China. In 2001, China was admitted to the World Trade Organization (WTO). Our leaders at that time argued that that decision would compel China to liberalize its economy and strengthen protections against things that were unacceptable to us and in support of private property and the rule of law."]


finetune_weights(question = questions_list, 
                 answer = responses_list,
                 neg_answer = neg_response_list,
                 question_encoder = question_encoder,
                 response_encoder = response_encoder,
                 neg_response_encoder = neg_response_encoder,
                 var_finetune = var_finetune,
                 optimizer = adam_optimizer)

