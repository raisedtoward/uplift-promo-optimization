from typing import Any, List, Dict, Tuple, Union
import re
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from .descriptive import seek_types, seek_categorical_from_integer_features


class CategoricalEncoder:
    def __init__(
        self,
        cat_na_fill_value: str = '0_MISSING_VALUE',
        map_unknown_to_na: bool = True,
        enforce_string: bool = False,
        verbose: bool = True,
        high_cardinality_threshold: int = 100,
        suffix: str = '_ENCODED',
        seek_categorical_from_int: bool = True,
    ):
        self.cat_na_fill_value = cat_na_fill_value
        self.map_unknown_to_na = map_unknown_to_na
        self.enforce_string = enforce_string
        self.verbose = verbose
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_features = []
        self.fitted = False
        self.category_to_index_transformers = None
        self.index_to_category_transformers = None
        self.suffix = suffix
        self.seek_categorical_from_int = seek_categorical_from_int

    @property
    def features_encoded(self):
        if self.fitted:
            return list(map(lambda s: f"{s}{self.suffix}", self.index_to_category_transformers.keys()))

    @features_encoded.setter
    def features_encoded(self, value):
        raise AttributeError('unable to set without reinitialization and refit')

    @features_encoded.deleter
    def features_encoded(self):
        raise AttributeError('cannot be deleted')

    def fit(
        self,
        data: pd.DataFrame,
        cat_features: List[str] = None,
    ) -> None:

        _ = self.fit_transform(data = data, cat_features = cat_features, transform_inplace = False)

    def fit_transform(
        self,
        data: pd.DataFrame,
        cat_features: List[str] = None,
        transform_inplace: bool = False,
        return_features: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:

        LE = LabelEncoder()
        if cat_features is None:
            cat_features, _, int_features, _, _, _ = seek_types(data)
            if self.seek_categorical_from_int:
                cat_features = cat_features + seek_categorical_from_integer_features(data, int_features)
            if self.verbose:
                print(f"The following features recognised as categorical: {cat_features}")
        df_cat_features = [col for col in data.columns if col in cat_features]
        if self.verbose:
            print(f"Categorical features in dataframe are: {df_cat_features}")

        if transform_inplace:
            df_cat = data[df_cat_features]
        else:
            df_cat = deepcopy(data[df_cat_features])
        ### during encoding nans are filled with a certain value
        df_cat = df_cat.fillna(self.cat_na_fill_value, inplace = False)
        if self.enforce_string:
            df_cat = df_cat.applymap(lambda x: str(x))

        category_to_index_transformers = []
        index_to_category_transformers = []

        for col in df_cat_features:
            mapped = LE.fit_transform(df_cat[col])
            if self.verbose:
                print('-'*50)
                print(col)
                print(f"Initial categories: {LE.classes_}")
            df_cat[col] = mapped
            index_to_category = dict(enumerate(LE.classes_))
            category_to_index = {v: k for k, v in index_to_category.items()}
            index_to_category_transformers.append(index_to_category)
            category_to_index_transformers.append(category_to_index)

        self.category_to_index_transformers = dict(zip(df_cat_features, category_to_index_transformers))
        self.index_to_category_transformers = dict(zip(df_cat_features, index_to_category_transformers))
        self.fitted = True

        if not transform_inplace:
            df_cat_features = [f"{f}{self.suffix}" for f in df_cat_features]
            df_cat.columns = df_cat_features
        if return_features:
            return df_cat, df_cat_features
        return df_cat

    def transform(
        self,
        data: pd.DataFrame,
        cat_features_for_transform: List[str] = None,
        transform_inplace: bool = False,
        return_features: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:

        if not self.fitted:
            raise ValueError("encoder should be fitted first")

        if cat_features_for_transform is None:
            df_cat_features = [col for col in data.columns if col in self.category_to_index_transformers.keys()]
        else:
            df_cat_features = [
                col for col in data.columns
                if col in self.category_to_index_transformers and col in cat_features_for_transform
            ]

        if self.verbose:
            print(f"The following features will be transformed: {df_cat_features}")
        if transform_inplace:
            df_cat = data[df_cat_features]
        else:
            df_cat = deepcopy(data[df_cat_features])
        ### when encoding, nans are filled with a certain value
        df_cat = df_cat.fillna(self.cat_na_fill_value, inplace = False)

        for col in df_cat_features:
            transformer = self.category_to_index_transformers[col]
            ### first filling Nans
            df_cat[col] = df_cat[col].fillna(self.cat_na_fill_value)

            known_categories = set(transformer.keys())
            current_categories = set(df_cat[col].unique())
            unknown_categories = current_categories.difference(known_categories)
            unk_mask = df_cat[col].isin(unknown_categories)

            if len(unknown_categories) > 0 and self.verbose:
                print('-'*50)
                print(col)
                print(
                    f"Previously unknown categories were met: {unknown_categories}\n"
                    f"They will be replaced with value for NA: '{self.cat_na_fill_value}'"
                )
            # for value in unknown_categories:
            #     if self.map_unknown_to_na:
            #         ### assigning code for missing value for all unknown categories
            #         df_cat.loc[df_cat[col] == value, col] = self.cat_na_fill_value
            #     else:
            #         # too complicated and no need for this
            #         pass
            ### if code for missing values already in dictionary then it's okay, if not - adding it to both transformers
            if self.cat_na_fill_value not in known_categories:
                new_index = max(transformer.values()) + 1
                transformer[self.cat_na_fill_value] = new_index

                self.category_to_index_transformers[col][self.cat_na_fill_value] = new_index
                self.index_to_category_transformers[col][new_index] = self.cat_na_fill_value

            ### assigning code for missing value for all unknown categories
            ### !order of these two calls matter!
            df_cat.loc[unk_mask, col] = self.cat_na_fill_value
            df_cat[col] = df_cat[col].replace(transformer, inplace = False)
            if self.enforce_string:
                df_cat[col] = df_cat[col].astype(str)
            else:
                df_cat[col] = df_cat[col].astype(int)

        if not transform_inplace:
            df_cat_features = [f"{f}{self.suffix}" for f in df_cat_features]
            df_cat.columns = df_cat_features
        if return_features:
            return df_cat, df_cat_features
        return df_cat


class HotEncoder:
    def __init__(
        self,
        force_nan_and_unknown_category: bool = True,
        map_unknown_to_na: bool = True,
        seek_categorical_from_int: bool = False,
        cat_na_fill_value: str = '(MISSING_OR_UNKNOWN_VALUE)',
        drop: str = 'first',
    ):

        """
        'drop' is either "first" or "last" or None
        when 'force_nan_and_unknown_category' is True,
        even if nans are not present in training set corresponding column would be added
        it would not be dropped either way and would not cause multicollinearity in linear models
        or carry any info as it's constantly 0,
        but it allows safely handling new unknown values or nans on inference
        
        when 'map_unknown_to_na' is True, on transform new values goes to 'cat_na_fill_value' first
        and then strategy depends on whether 'force_nan_and_unknown_category' was set to True
        if not, it would raise an error
        """

        self.force_nan_and_unknown_category = force_nan_and_unknown_category
        self.map_unknown_to_na = map_unknown_to_na
        self.cat_na_fill_value = cat_na_fill_value
        if not (drop is None or drop in ['first', 'last', 'na']):
            raise ValueError(f"'drop' argument must be one of the {['first', 'last', 'na']}")
        self.drop = drop
        self.sparse = False
        self.verbose = False
        self.seek_categorical_from_int = seek_categorical_from_int
        self.transformers = None

    def fit(
        self,
        data: pd.DataFrame,
        cat_features: List[str],
    ):

        if cat_features is None:
            cat_features, _, int_features, _, _, _ = seek_types(data)
            if self.seek_categorical_from_int:
                cat_features = cat_features + seek_categorical_from_integer_features(data, int_features)
            if self.verbose:
                print(f"The following features recognised as categorical: {cat_features}")
        df_cat_features = [col for col in data.columns if col in cat_features]
        if self.verbose:
            print(f"Categorical features in dataframe are: {df_cat_features}")

        transformers = defaultdict(lambda: defaultdict(object))

        for cat_feature in df_cat_features:

            if any(data[cat_feature].isna()) or self.force_nan_and_unknown_category:
                fit_categories = pd.concat([
                    data[cat_feature].fillna(self.cat_na_fill_value), pd.Series(self.cat_na_fill_value)
                ], ignore_index = True)
            else:
                fit_categories = data[cat_feature].copy()

            ### if feature isn't str-like but has missing values or self.force_nan_and_unknown_category is True
            ### we end up with mixed-type column and OHE wouldn't work
            str_mapper = {v: str(v) for v in set(fit_categories)}
            ohe = OneHotEncoder(
                drop = None,
                handle_unknown = 'ignore' if self.map_unknown_to_na else 'error',
                sparse = self.sparse
            )
            ohe.fit(fit_categories.map(str_mapper)[:, None])
            transformers[cat_feature]['transformer'] = ohe
            transformers[cat_feature]['categories'] = list(str_mapper.keys())
            transformers[cat_feature]['str_mapper'] = str_mapper

        self.transformers = {k: dict(v) for k, v in transformers.items()}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        X_out = []
        for cat_feature in self.transformers:
            transformer = self.transformers[cat_feature]['transformer']
            str_mapper = self.transformers[cat_feature]['str_mapper']
            categories_ordered = list(transformer.categories_[0])  # [0] is a first and single array in a list

            x = data[cat_feature].fillna(self.cat_na_fill_value).map(str_mapper)
            if self.map_unknown_to_na:
                x = x.fillna(self.cat_na_fill_value)
            X = transformer.transform(x[:, None])
            X = pd.DataFrame(X, columns = categories_ordered, index = data.index)

            if self.drop is None:
                drop = []
            else:
                drop = []
                if self.drop == 'na' and self.cat_na_fill_value in self.transformers:
                    drop = [self.cat_na_fill_value]
                else:
                    if self.map_unknown_to_na:
                        non_na_cols = [f for f in X.columns]  # if f != self.cat_na_fill_value]
                    else:
                        non_na_cols = [f for f in X.columns if f != self.cat_na_fill_value]
                    if self.drop == 'first':
                        drop = [non_na_cols[0]]
                    elif self.drop == 'last':
                        drop = [non_na_cols[-1]]

            X.drop(drop, axis = 1, inplace = True)
            X.columns = [cat_feature + '__' + re.sub('\s+', '_', f) for f in X.columns]
            X_out.append(X)

        if X_out:
            return pd.concat(X_out, axis = 1)
        else:
            return pd.DataFrame()

    def fit_transform(
        self,
        data: pd.DataFrame,
        cat_features: List[str],
    ):
        self.fit(data = data, cat_features = cat_features)
        return self.transform(data)
