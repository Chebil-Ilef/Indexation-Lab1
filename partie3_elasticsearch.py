"""
Partie 3: Indexation avec Elasticsearch

Ce script:
1. Crée un index Elasticsearch avec un analyzer personnalisé
2. Visualise et commente _analyze, _segments, _stats
3. Mesure le temps d'indexation avec 1, 2, 4 shards
4. Mesure la taille disque avant/après _forcemerge
5. Compare avec l'indexation manuelle en Python
6. Discute de la gestion de la compression, maintenance et parallélisation
"""

import time
import json
import os
from typing import Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from preprocess_build_index import corpus, get_preprocessed_corpus, build_inverted_index
from metrics import format_bytes


# Configuration Elasticsearch
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ES_URL = f"http://{ES_HOST}:{ES_PORT}"

# Connexion à Elasticsearch
es = Elasticsearch([ES_URL], request_timeout=60)


def to_dict(obj):
    """
    Convertit une réponse Elasticsearch en dictionnaire Python.
    Gère les ObjectApiResponse et autres types de réponses.
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, 'body'):
        return obj.body
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, 'items'):
        return {k: to_dict(v) for k, v in obj.items()}
    else:
        return obj


def check_elasticsearch_connection():
    """Vérifie la connexion à Elasticsearch"""
    try:
        if es.ping():
            print(f"✓ Connexion Elasticsearch réussie: {ES_URL}")
            print(f"  Version: {es.info()['version']['number']}")
            return True
        else:
            print("✗ Impossible de se connecter à Elasticsearch")
            return False
    except Exception as e:
        print(f"✗ Erreur de connexion à Elasticsearch: {e}")
        print(f"  Assurez-vous qu'Elasticsearch est démarré sur {ES_URL}")
        return False


def create_index_with_custom_analyzer(index_name: str, num_shards: int = 1):
    """
    Crée un index Elasticsearch avec un analyzer personnalisé
    qui reproduit le preprocessing Python:
    - lowercase
    - remove punctuation
    - remove stopwords
    - stemming (approximation de lemmatization)
    """
    
    # Supprimer l'index s'il existe
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"  Index '{index_name}' supprimé")
    
    # Définition de l'analyzer personnalisé
    # Note: Elasticsearch n'a pas de lemmatization native comme spaCy,
    # on utilise le stemming comme approximation
    index_settings = {
        "settings": {
            "number_of_shards": num_shards,
            "number_of_replicas": 0,  # Pas de réplicas pour les tests
            "analysis": {
                "analyzer": {
                    "custom_corpus_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",                    # Unification de la casse
                            "asciifolding",                 # Suppression des accents
                            "punctuation_remover",          # Suppression de la ponctuation
                            "stop",                         # Suppression des stopwords
                            "english_stemmer"               # Stemming (approximation lemmatization)
                        ]
                    }
                },
                "filter": {
                    "punctuation_remover": {
                        "type": "pattern_replace",
                        "pattern": "[^a-z0-9]",
                        "replacement": ""
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "custom_corpus_analyzer",
                    "search_analyzer": "custom_corpus_analyzer"
                },
                "doc_id": {
                    "type": "integer"
                }
            }
        }
    }
    
    # Création de l'index
    es.indices.create(index=index_name, body=index_settings)
    print(f"✓ Index '{index_name}' créé avec {num_shards} shard(s)")
    
    return index_name


def test_analyze(index_name: str, sample_text: str):
    """
    Teste l'analyzer personnalisé avec _analyze
    """
    print(f"\n{'='*60}")
    print("1. TEST _analyze")
    print(f"{'='*60}")
    print(f"Texte d'exemple: '{sample_text}'")
    
    # Analyse avec l'analyzer personnalisé
    result = es.indices.analyze(
        index=index_name,
        body={
            "analyzer": "custom_corpus_analyzer",
            "text": sample_text
        }
    )
    
    # Convertir le résultat en dictionnaire (ObjectApiResponse -> dict)
    result_dict = to_dict(result)
    
    print("\nRésultat de l'analyse:")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    
    tokens = [token["token"] for token in result_dict["tokens"]]
    print(f"\nTokens extraits: {tokens}")
    
    # Comparaison avec le preprocessing Python
    from preprocess_build_index import preprocess
    python_tokens = preprocess(sample_text)
    print(f"Tokens Python:   {python_tokens}")
    
    print("\nCommentaire:")
    print("- L'analyzer Elasticsearch applique les mêmes transformations de base")
    print("- Le stemming est une approximation de la lemmatization (ex: 'documents' -> 'document')")
    print("- Les résultats peuvent différer légèrement de spaCy mais restent cohérents")


def visualize_segments(index_name: str):
    """
    Visualise le contenu de _segments
    """
    print(f"\n{'='*60}")
    print("2. VISUALISATION _segments")
    print(f"{'='*60}")
    
    segments = es.indices.segments(index=index_name)
    
    # Convertir en dictionnaire si nécessaire
    segments_dict = to_dict(segments)
    
    print("\nSegments de l'index:")
    print(json.dumps(segments_dict, indent=2, ensure_ascii=False))
    
    # Analyse des segments
    shards = segments_dict["indices"][index_name]["shards"]
    total_segments = 0
    total_size = 0
    
    for shard_num, shard_data in shards.items():
        for shard_info in shard_data:
            segments_dict = shard_info.get("segments", {})
            num_segments = len(segments_dict)
            total_segments += num_segments
            
            # Sommer les tailles de tous les segments dans ce shard
            shard_size = sum(
                segment_info.get("size_in_bytes", 0)
                for segment_info in segments_dict.values()
            )
            total_size += shard_size
            
            print(f"\nShard {shard_num}:")
            print(f"  Nombre de segments: {num_segments}")
            print(f"  Taille: {format_bytes(shard_size)}")
    
    print(f"\nTotal:")
    print(f"  Segments: {total_segments}")
    print(f"  Taille: {format_bytes(total_size)}")
    
    print("\nCommentaire:")
    print("- Les segments sont les unités de stockage d'Elasticsearch")
    print("- Chaque segment est un index inversé immutable")
    print("- Plus de segments = plus de fichiers à gérer, mais meilleure parallélisation")
    print("- _forcemerge réduit le nombre de segments pour optimiser les performances")


def visualize_stats(index_name: str):
    """
    Visualise les statistiques de _stats
    """
    print(f"\n{'='*60}")
    print("3. STATISTIQUES _stats")
    print(f"{'='*60}")
    
    stats = es.indices.stats(index=index_name)
    stats_dict = to_dict(stats)
    
    index_stats = stats_dict["indices"][index_name]
    
    print("\nStatistiques principales:")
    print(json.dumps({
        "total": {
            "docs": {
                "count": index_stats["total"]["docs"]["count"],
                "deleted": index_stats["total"]["docs"]["deleted"]
            },
            "store": {
                "size_in_bytes": index_stats["total"]["store"]["size_in_bytes"]
            },
            "indexing": {
                "index_total": index_stats["total"]["indexing"]["index_total"],
                "index_time_in_millis": index_stats["total"]["indexing"]["index_time_in_millis"]
            }
        }
    }, indent=2))
    
    # Statistiques détaillées
    total = index_stats["total"]
    print(f"\nRésumé:")
    print(f"  Documents indexés: {total['docs']['count']}")
    print(f"  Documents supprimés: {total['docs']['deleted']}")
    print(f"  Taille disque: {format_bytes(total['store']['size_in_bytes'])}")
    print(f"  Opérations d'indexation: {total['indexing']['index_total']}")
    print(f"  Temps total d'indexation: {total['indexing']['index_time_in_millis']} ms")
    
    print("\nCommentaire:")
    print("- _stats fournit des métriques détaillées sur l'index")
    print("- Permet de suivre la performance, la taille, et l'utilisation des ressources")
    print("- Utile pour le monitoring et l'optimisation")


def index_documents(index_name: str, documents: list):
    """
    Indexe les documents dans Elasticsearch
    """
    actions = [
        {
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "doc_id": doc_id,
                "content": doc
            }
        }
        for doc_id, doc in enumerate(documents)
    ]
    
    bulk(es, actions)
    es.indices.refresh(index=index_name)
    
    return len(documents)


def index_documents_with_multiple_segments(index_name: str, documents: list, batch_size: int = 3):
    """
    Indexe les documents en plusieurs petits batches pour créer plusieurs segments.
    Chaque batch est indexé séparément avec un refresh, ce qui crée un nouveau segment.
    """
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        actions = [
            {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "doc_id": doc_id,
                    "content": doc
                }
            }
            for doc_id, doc in enumerate(batch, start=i)
        ]
        
        bulk(es, actions)
        # Refresh après chaque batch pour créer un nouveau segment
        es.indices.refresh(index=index_name)
    
    return total_docs


def measure_indexing_time(num_shards: int, documents: list):
    """
    Mesure le temps d'indexation avec un nombre donné de shards
    """
    index_name = f"corpus_shards_{num_shards}"
    
    # Créer l'index
    create_index_with_custom_analyzer(index_name, num_shards)
    
    # Mesurer le temps d'indexation
    # Utiliser indexation par batches pour créer plusieurs segments
    t0 = time.perf_counter()
    num_docs = index_documents_with_multiple_segments(index_name, documents, batch_size=3)
    t1 = time.perf_counter()
    
    indexing_time = t1 - t0
    
    # Obtenir la taille disque
    stats = es.indices.stats(index=index_name)
    stats_dict = to_dict(stats)
    disk_size = stats_dict["indices"][index_name]["total"]["store"]["size_in_bytes"]
    
    return {
        "num_shards": num_shards,
        "indexing_time": indexing_time,
        "disk_size": disk_size,
        "index_name": index_name
    }


def measure_forcemerge_impact(index_name: str):
    """
    Mesure l'impact de _forcemerge sur la taille disque
    """
    # Taille avant forcemerge
    stats_before = es.indices.stats(index=index_name)
    stats_before_dict = to_dict(stats_before)
    size_before = stats_before_dict["indices"][index_name]["total"]["store"]["size_in_bytes"]
    segments_before = es.indices.segments(index=index_name)
    segments_before_dict = to_dict(segments_before)
    num_segments_before = sum(
        len(shard_info.get("segments", {}))
        for shard_data in segments_before_dict["indices"][index_name]["shards"].values()
        for shard_info in shard_data
    )
    
    # Forcemerge
    print(f"\n  Exécution de _forcemerge...")
    es.indices.forcemerge(index=index_name, max_num_segments=1, wait_for_completion=True)
    
    # Attendre que le merge soit terminé et rafraîchir
    es.indices.refresh(index=index_name)
    
    # Taille après forcemerge
    stats_after = es.indices.stats(index=index_name)
    stats_after_dict = to_dict(stats_after)
    size_after = stats_after_dict["indices"][index_name]["total"]["store"]["size_in_bytes"]
    segments_after = es.indices.segments(index=index_name)
    segments_after_dict = to_dict(segments_after)
    num_segments_after = sum(
        len(shard_info.get("segments", {}))
        for shard_data in segments_after_dict["indices"][index_name]["shards"].values()
        for shard_info in shard_data
    )
    
    return {
        "size_before": size_before,
        "size_after": size_after,
        "num_segments_before": num_segments_before,
        "num_segments_after": num_segments_after
    }


def compare_with_manual_indexing(documents: list):
    """
    Compare les résultats avec l'indexation manuelle en Python
    """
    print(f"\n{'='*60}")
    print("4. COMPARAISON AVEC INDEXATION MANUELLE")
    print(f"{'='*60}")
    
    # Indexation manuelle
    preprocessed_corpus = get_preprocessed_corpus()
    
    t0 = time.perf_counter()
    manual_index = build_inverted_index(preprocessed_corpus)
    t1 = time.perf_counter()
    manual_time = t1 - t0
    
    # Taille mémoire de l'index manuel
    from metrics import deep_size
    manual_size = deep_size(manual_index)
    
    print(f"\nIndexation manuelle Python:")
    print(f"  Temps: {manual_time:.6f} secondes")
    print(f"  Taille mémoire: {format_bytes(manual_size)}")
    print(f"  Nombre de termes: {len(manual_index)}")
    
    # Indexation Elasticsearch (1 shard pour comparaison)
    es_result = measure_indexing_time(1, documents)
    
    print(f"\nIndexation Elasticsearch (1 shard):")
    print(f"  Temps: {es_result['indexing_time']:.6f} secondes")
    print(f"  Taille disque: {format_bytes(es_result['disk_size'])}")
    
    # Comparaison
    print(f"\nComparaison:")
    print(f"  Ratio temps (ES/Manuel): {es_result['indexing_time'] / manual_time:.2f}x")
    print(f"  Ratio taille (ES/Manuel): {es_result['disk_size'] / manual_size:.2f}x")
    
    return {
        "manual_time": manual_time,
        "manual_size": manual_size,
        "es_time": es_result['indexing_time'],
        "es_size": es_result['disk_size']
    }


def main():
    """
    Fonction principale pour Partie 3
    """
    print("="*60)
    print("PARTIE 3: INDEXATION AVEC ELASTICSEARCH")
    print("="*60)
    
    # Vérifier la connexion
    if not check_elasticsearch_connection():
        return
    
    # Créer un index de test pour les analyses
    test_index = "corpus_test"
    create_index_with_custom_analyzer(test_index, num_shards=1)
    
    # Indexer quelques documents pour les tests
    index_documents(test_index, corpus)
    
    # 1. Tester _analyze
    test_analyze(test_index, corpus[0])
    
    # 2. Visualiser _segments
    visualize_segments(test_index)
    
    # 3. Visualiser _stats
    visualize_stats(test_index)
    
    # 4. Mesurer le temps d'indexation avec différents nombres de shards
    print(f"\n{'='*60}")
    print("5. MESURE DU TEMPS D'INDEXATION (1, 2, 4 SHARDS)")
    print(f"{'='*60}")
    
    results = []
    for num_shards in [1, 2, 4]:
        print(f"\nTest avec {num_shards} shard(s)...")
        result = measure_indexing_time(num_shards, corpus)
        results.append(result)
        print(f"  Temps: {result['indexing_time']:.6f} secondes")
        print(f"  Taille: {format_bytes(result['disk_size'])}")
    
    # Tableau récapitulatif
    print(f"\n{'Shards':<10} {'Temps (s)':<15} {'Taille':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['num_shards']:<10} {r['indexing_time']:<15.6f} {format_bytes(r['disk_size']):<15}")
    
    # 5. Mesurer l'impact de _forcemerge
    print(f"\n{'='*60}")
    print("6. MESURE AVANT/APRÈS _forcemerge")
    print(f"{'='*60}")
    
    for result in results:
        index_name = result['index_name']
        print(f"\nIndex: {index_name} ({result['num_shards']} shard(s))")
        
        merge_result = measure_forcemerge_impact(index_name)
        
        print(f"  Avant forcemerge:")
        print(f"    Segments: {merge_result['num_segments_before']}")
        print(f"    Taille: {format_bytes(merge_result['size_before'])}")
        print(f"  Après forcemerge:")
        print(f"    Segments: {merge_result['num_segments_after']}")
        print(f"    Taille: {format_bytes(merge_result['size_after'])}")
        print(f"  Réduction: {format_bytes(merge_result['size_before'] - merge_result['size_after'])} "
              f"({(1 - merge_result['size_after']/merge_result['size_before'])*100:.1f}%)")
    
    # 6. Comparer avec l'indexation manuelle
    compare_with_manual_indexing(corpus)
    
    # 7. Discussion
    print(f"\n{'='*60}")
    print("7. DISCUSSION: ELASTICSEARCH VS IMPLÉMENTATION MANUELLE")
    print(f"{'='*60}")
    
    print("""
COMPRESSION:
- Elasticsearch utilise automatiquement la compression LZ4 ou DEFLATE
- Les segments sont compressés au niveau du système de fichiers
- Compression transparente sans intervention manuelle
- Notre implémentation manuelle nécessite GAP encoding + VByte explicitement

MAINTENANCE:
- Elasticsearch gère automatiquement la fusion de segments (merge)
- Support natif pour l'ajout/suppression de documents
- Optimisation automatique avec _forcemerge
- Notre implémentation nécessite une gestion manuelle des opérations CRUD

PARALLÉLISATION:
- Elasticsearch parallélise naturellement avec les shards
- Chaque shard peut être traité indépendamment
- Distribution possible sur plusieurs nœuds
- Notre implémentation utilise multiprocessing mais reste limitée à un seul nœud

AVANTAGES ELASTICSEARCH:
✓ Gestion automatique de la compression, maintenance, et parallélisation
✓ Scalabilité horizontale (multi-nœuds)
✓ Optimisations avancées (caching, query optimization)
✓ Support de fonctionnalités avancées (faceting, aggregations, etc.)

AVANTAGES IMPLÉMENTATION MANUELLE:
✓ Contrôle total sur les algorithmes
✓ Pas de dépendance externe
✓ Compréhension approfondie des mécanismes internes
✓ Légèreté pour de petits corpus
    """)
    
    # Nettoyage
    print(f"\n{'='*60}")
    print("NETTOYAGE")
    print(f"{'='*60}")
    
    indices_to_delete = ["corpus_test"] + [r['index_name'] for r in results]
    for index_name in indices_to_delete:
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"  Index '{index_name}' supprimé")
    
    print("\n✓ Partie 3 terminée!")


if __name__ == "__main__":
    main()

