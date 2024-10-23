class SCPruning:

    def __init__(self, ruleset, alpha, df) -> None:
        self.ruleset = ruleset # list
        self.alpha = alpha # float
        self.df = df # pandas.core.frame.DataFrame

    def pruning(self):
        pruned_ruleset = []
        df_copy = self.df.copy()
        found_rule = True

        while (len(self.ruleset) > 0 and not df_copy.empty and found_rule):
            cov = [self.compute_coverage(rule, df_copy) for rule in self.ruleset]
            conf = [self.compute_confidence(rule, df_copy) for rule in self.ruleset]
            sorted_ruleset = self.sorted_rules(conf, cov)
            found_rule = False

            # for rule, conf, cov in sorted_ruleset:
            #     if conf > self.alpha:
            #         df_copy = self.delete_data(rule, df_copy)
            #         self.ruleset.remove(rule)
            #         pruned_ruleset.append(rule)
            #         found_rule = True
            #         break
            break
        print("completed")
        return pruned_ruleset
    

    def compute_coverage(self, rule, df):
        pass

    def compute_confidence(self, rule, df):
        pass

    def sorted_rules(self, conf, cov):
        pass

    def delete_data(self, rule, df):
        pass