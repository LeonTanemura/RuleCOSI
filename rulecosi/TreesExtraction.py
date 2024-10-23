class XGBTreeExtraction:
    """
    A class used to extract and handle trees from an XGBoost model.

    Attributes:
        model (XGBoostClassifier) : The XGBoost model from which trees are extracted.
        tree (list): A list containing the dumped representation of the trees in the model.

    Methods:
        get_tree(): Returns the dumped representation of the trees.
    """
    def __init__(self, model) -> None:
        self.model = model
        self.booster = model.get_booster()
    
    def get_tree(self):
        tree = self.booster.get_dump()
        boost_round = self.booster.num_boosted_rounds()
        df = self.booster.trees_to_dataframe()
        return tree, boost_round, df
