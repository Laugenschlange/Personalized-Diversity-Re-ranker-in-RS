# Personalized-Diversity-Re-ranker-in-RS
this is the repo for saving the codes of the thesis 'Adapting Personalized Diversity from Commercial to Research Paper Recommendations'
### Dependencies:
   * Tensorflow 1.x
   * numpy
   * sklearn 

### Dataset
   * citeulike-a: https://github.com/js05212/citeulike-a
   * MovieLens-20M: https://files.grouplens.org/datasets/movielens/ml-20m-README.html 

### Use
   * reproduce result of RAPID on MovieLens-20M:
      * ```python
        python working_code/preprocess_ml.py
        ```
      * ```python
        python working_code/run_init_ranker_ml.py
        ```
      * ```python
        python working_code/run_test.py
        ```
 * apply RAPID on CiteULike-a:
    * ```python
       python working_code/preprocess_citeulike.py
       ```
    * ```python
       python working_code/run_init_ranker.py
       ```
    * ```python
       python working_code/run_citeulike.py
       ```
