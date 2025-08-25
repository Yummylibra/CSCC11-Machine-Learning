import numpy as np

class PolynomialRegression:
    def __init__(self, K, l2_coeff=0):
        """ Initialize the class of the polynomial regression model. 
            For given a single scalar input x:
            f(x) = b + w_1 * x + w_2 * x^2 + ... + w_K * x^K

        args:
            - K (int): The degree of the polynomial. Note: 1 <= K <= 15
            - parameters (ndarray (shape: (K + 1, 1))): The model parameters.
            - l2_coeff (float): The coefficient of the L2 regularization term.

        The bias term is the first element in self.parameters (i.e. self.parameters = [b, w_1, ..., w_K]^T).
        """
        assert 1 <= K <= 15, f"K must be between 1 and 15. Got: {K}"
        assert l2_coeff >= 0, f"l2_coeff must be non-negative. Got: {l2_coeff}"

        self.K = K
        self.l2_coeff = l2_coeff
        self.parameters = np.ones((K + 1, 1), dtype=np.float32)

    def predict(self, X):
        """ evaluates the polynomial model at N input data points. 

        args:
            - X (ndarray (shape: (N, 1))): A column vector consisting N scalar input data.
        output:
            - prediction (ndarray (shape: (N, 1))): A column vector consisting N scalar output data. 
        
        """
        assert X.shape == (X.shape[0], 1)
        
        # ====================================================
        # return the output of the polynomial function
        features = np.vander(X.flatten(), N = self.parameters.size, increasing = True)
        # [1,x1,x1^2,x1^3...x1^k]
        # [1,x2,x2^2,x2^3...x2^k]
        # [         ...         ]
        # [1,xn,xn^2,xn^3...xn^k]
        prediction = features.dot(self.parameters)
        # ====================================================
        return prediction
    
    def fit(self, train_X, train_Y):
        """ This method fits the model parameters, given the training inputs and outputs.
            This method does not have output.

        args:
            - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
            - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0], 1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."
        assert train_X.shape[0] >= self.K, f"require more data points to fit a polynomial (train_X: {train_X.shape}, K: {self.K}). Do you know why?"

        # ===========================================================================
        #  Set self.parameters to least square solution for the polynomial basis
        B = np.vander(train_X.flatten(), N=self.K + 1, increasing=True)
        C = np.linalg.inv(B.T.dot(B)).dot(B.T)#C = B+

        self.parameters = C.dot(train_Y)
        # ===========================================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def fit_with_l2_regularization(self, train_X, train_Y):
        """ This method fits the model parameters with L2 regularization, given the training inputs and outputs.
        This method does not have output.

        args:
            - train_X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training inputs.
            - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
            - l2_coef (float): The lambda term that decides how much regularization we want.
            
  
        """
        assert train_X.shape == train_Y.shape and train_X.shape == (train_X.shape[0], 1), f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_Y: {train_Y.shape})."

        # =======================================================================================
        # TODO: Set self.parameters to regularized least square solution for the polynomial basis
        B = np.vander(train_X.flatten(), N=self.K + 1, increasing=True)
        B_transpose = B.T
        B_transpose_B = B_transpose.dot(B)
        I = np.eye(B_transpose_B.shape[0])
        regularization_term = self.l2_coeff * I
        C = np.linalg.inv(B_transpose_B + regularization_term).dot(B_transpose)
    
        self.parameters = C.dot(train_Y)
        # =======================================================================================

        assert self.parameters.shape == (self.K + 1, 1)

    def compute_mse(self, X, observed_Y):
        """ This method computes the mean squared error.

        args:
            - X (ndarray (shape: (N, 1))): A N-column vector consisting N scalar inputs.
            - observed_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        output:
            - mse (float): The mean squared error between the predicted Y and the observed Y.
        """
        assert X.shape == observed_Y.shape and X.shape == (X.shape[0], 1), f"input and/or output has incorrect shape (X: {X.shape}, observed_Y: {observed_Y.shape})."

        # ========================================================================================
        # TODO: Run prediction over X and compute the mean squared error with the observed outputs

        
        predicted_Y = self.predict(X)
        errors = observed_Y - predicted_Y
        mse = np.mean(errors**2)
        # ========================================================================================
        
        return mse


if __name__ == "__main__":
    # You can use linear regression to check whether your implementation is correct.
    # NOTE: This is just a quick check but does not cover all cases.
    model = PolynomialRegression(K=1)
    train_X = np.expand_dims(np.arange(10), axis=1)
    train_Y = np.expand_dims(np.arange(10), axis=1)

    model.fit(train_X, train_Y)
    optimal_parameters = np.array([[0.], [1.]])
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Change parameters to suboptimal for next test
    model.parameters += 1
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[0.0231303], [0.99460293]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))