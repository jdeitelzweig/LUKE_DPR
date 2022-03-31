from collections import defaultdict
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from dpr.models import init_biencoder_components
from dpr.models.biencoder import dot_product_scores
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.model_utils import setup_for_distributed_mode, load_states_from_checkpoint
from dense_retriever import generate_question_vectors, validate, save_results
from generate_dense_embeddings import gen_ctx_vectors

logger = logging.getLogger()
setup_logger(logger)


def create_title_cache(passages):
    cache = defaultdict(list)
    for passage_id, passage in passages:
        cache[passage.title].append((passage_id, passage))
    return cache


def find_relevant_passages(entities, title_cache):
    relevant = []
    for ent in entities:
        relevant.extend(title_cache[ent])
    return relevant


@hydra.main(config_path="conf", config_name="drl")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    question_encoder = encoder.question_model
    ctx_encoder = encoder.ctx_model

    question_encoder, _ = setup_for_distributed_mode(question_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    ctx_encoder, _ = setup_for_distributed_mode(ctx_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    question_encoder.eval()
    ctx_encoder.eval()

    # get questions & answers
    questions = []
    question_answers = []
    question_entities = []
    question_spans = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    # filter out questions that don't have entities
    for qa_sample in qa_src:
        question, answers, entities, spans = qa_sample.query, qa_sample.answers, qa_sample.entities, qa_sample.entity_spans
        if entities:
            questions.append(question)
            question_answers.append(answers)
            question_entities.append(entities)
            question_spans.append(spans)

    logger.info("questions len %d", len(questions))

    if qa_src.selector:
        logger.info("Using custom representation token selector")

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = generate_question_vectors(
        question_encoder,
        tensorizer,
        questions,
        question_entities,
        question_spans,
        cfg.batch_size,
        query_token=qa_src.special_query_token,
        selector=qa_src.selector,
    )

    # load contexts
    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]
    logger.info("Caching contexts by titles")
    passages_by_title = create_title_cache(all_passages)

    # for each question, find contexts with title associated with entities
    top_results_and_scores = []
    logger.info("Getting results")
    for i, (question, entities) in enumerate(zip(questions, question_entities)):
        q_vector = questions_tensor[i]
        relevant_passages = find_relevant_passages(entities, passages_by_title)
        # encode passages
        data = gen_ctx_vectors(cfg, relevant_passages, ctx_encoder, tensorizer, True)
        
        if len(data) == 0:
            top_results_and_scores.append(([], []))
            continue
        
        ctx_ids = []
        ctx_vectors = torch.empty((len(data), len(data[0][1])))
        for j, (id, vector) in enumerate(data):
            ctx_ids.append(id)
            ctx_vectors[j] = torch.tensor(vector)

        # rank by similarity scores
        scores = dot_product_scores(q_vector, ctx_vectors)

        sorted_ids, sorted_scores = (list(t) for t in zip(*sorted(zip(ctx_ids, scores.tolist()), key=lambda x: x[1], reverse=True)))

        top_results_and_scores.append((sorted_ids[:cfg.n_docs], sorted_scores[:cfg.n_docs]))

        if i % 100 == 0:
            logger.info(f"Retrieved passages for questions {i}/{len(questions)}")

    # validate
    questions_doc_hits = validate(
        all_passages_dict,
        question_answers,
        top_results_and_scores,
        cfg.validation_workers,
        cfg.match,
    )

    if cfg.out_file:
        save_results(
            all_passages_dict,
            questions,
            question_answers,
            top_results_and_scores,
            questions_doc_hits,
            cfg.out_file,
        )


if __name__ == "__main__":
    main()
