# #21-cp-26
import pandas as pd
# # a. Create a Pandas DataFrame from a dictionary.
# data = {'Name': ['Ibrahim', 'Ahsan' ,'minahil'],
#         'Age': [27, 00, 26],
#         'City': ['New York', 'San Andreas', 'tokss']}
# df = pd.DataFrame(data)
#
# # b. Create a Pandas Series from a list.
# heights = [175, 160, 180]
# heights_series = pd.Series(heights, name='Height')
#
# # c. Access a specific column in a DataFrame.
# ages_column = df['Age']
#
# # d. Add a new column to an existing DataFrame.
# df['Gender'] = ['Male', 'male', 'female']
# print("DataFrame:")
# print(df)
# print("\nSeries:")
# print(heights_series)
# print("\nAccessed specific column 'Age' in DataFrame:")
# print(ages_column)
#














#
# import pandas as pd
# data = {'Name': ['EBERA', 'ABRA', 'IBRA', 'SHIBRA', 'LINCON'],
#         'Age': [25, None, 22, 30, 25],
#         'City': ['ISB', 'San', 'SGD', 'BHWL', 'ISB']}
# df = pd.DataFrame(data)
#
# # a Handle missing values in a DataFrame.
# df_filled = df.fillna({'Age': df['Age'].mean()})  # FIllING by mean of the column
#
# # b Remove duplicate rows from a DataFrame.
# df_no_duplicates = df.drop_duplicates()
#
# # c Rename columns in a DataFrame.
# df_renamed = df.rename(columns={'Name': 'Full Name', 'Age': 'Years'})
#
# print("Original DataFrame:")
# print(df)
# print("\nDataFrame with missing values handled:")
# print(df_filled)
# print("\nDataFrame with duplicate rows removed:")
# print(df_no_duplicates)
# print("\nDataFrame with columns renamed:")
# print(df_renamed)
#
# import numpy as np
# # a 1D and 2D NumPy array.
# array_1d = np.array([1, 2, 3, 4, 5])
# array_2d = np.array([[1, 2, 3], [4, 5, 6]])
#
# # b Reshape an array.
# reshaped_array = array_1d.reshape(5, 1)
#
# # c Slice and index arrays.
# sliced_array = array_1d[1:4]
# indexed_element = array_2d[1, 2]
#
# # d Perform element-wise operations (addition, subtraction, multiplication, etc.) on arrays.
# array_addition = array_1d + 2
# array_multiplication = array_2d * 3
# print("1D Array:")
# print(array_1d)
# print("\n2D Array:")
# print(array_2d)
# print("\nReshaped Array:")
# print(reshaped_array)
# print("\nSliced Array:")
# print(sliced_array)
# print("\nIndexed Element:")
# print(indexed_element)
# print("\nArray Addition:")
# print(array_addition)
# print("\nArray Multiplication:")
# print(array_multiplication)
# # sum ,mean function   part e
# array_sum = np.sum(array_2d)
# array_mean = np.mean(array_2d)
# array_max = np.max(array_1d)
#
# # f matrix multiplication.
# matrix_a = np.array([[1, 2], [3, 4]])
# matrix_b = np.array([[5, 6], [7, 8]])
# matrix_multiplication = np.dot(matrix_a, matrix_b)
#
# # g. mean, median tandard deviation of an array.
# array_mean_value = np.mean(array_1d)
# array_median_value = np.median(array_1d)
# array_std_deviation = np.std(array_1d)
#
# # h.  correlation and covariance b/w  arrays.
# array_x = np.array([1, 2, 3, 4, 5])
# array_y = np.array([5, 4, 3, 2, 1])
# correlation = np.corrcoef(array_x, array_y)
# covariance = np.cov(array_x, array_y)
# print("\nArray Sum:")
# print(array_sum)
# print("\nArray Mean:")
# print(array_mean)
# print("\nMaximum Value in Array:")
# print(array_max)
# print("\nMatrix Multiplication:")
# print(matrix_multiplication)
# print("\nArray Mean Value:")
# print(array_mean_value)
# print("\nArray Median Value:")
# print(array_median_value)
# print("\nArray Standard Deviation:")
# print(array_std_deviation)
# print("\nCorrelation Matrix:")
# print(correlation)
# print("\nCovariance Matrix:")
# print(covariance)
#21-cp-26
import numpy
arr = numpy.array([1, 2, 3, 4, 5])
print(arr)


# arr=numpy.array([1,2,3,4,5])
# print(arr)
# print(np.__version__)
# #exp 3
# print(type(arr))
# #exp 4
# arr=np.array((1,2,3,4,5))
# print("exp 4")
# print(arr)
# #exp5
# print(arr[0])
#
# #test your skilss
# print(arr[2])
#
# b=arr[2]
# c=arr[3]
# print(b+c)

# #exp6
# import numpy as np
# arr2=np.array([[1,2,3,4,5],[6,7,8,9,10]])
# print('2nd elemnet on 1st row',arr2[0,1])
#
# #exp 7
# arr3= np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print('Access the third element of the second array of the first array:', arr3[0,1,2])
#
# #exp 8
# print('Last element from 2nd dim:',arr2[1,-1])
#
# #exp 9 array slicing
# # import numpy as np
# arr=np.array([1,2,3,4,5,6,7])
# print(arr[1:5])
# import numpy as np
# arr=np.array([1,2,3,4,5,6,7])
# #test your skills
#
# print(arr[4:])
# print('start to 4 not include',arr[:4])

# #exp 10
# import numpy as np
# arr=np.array([1,2,3,4])
# print(arr.dtype)
# #exp 11
# arr=np.array(['apple','banana'])
# print(arr.dtype)
# #exp 12
# arr=np.array([1,2,3,4,5])
# x=arr.copy()
# arr[0]=42
# print(arr)
# print(x)
# #exp 13
# arr=np.array([1,2,3,4,5])
# x=arr.view()
# arr[0]=42
# print(arr)
# print(x)
# #exp 14 sorting
# arr=np.array([3,2,0,1])
# print(np.sort(arr))
# #exp 15
# arr = np.array(['banana', 'cherry', 'apple'])
# print(np.sort(arr))
#
# #exp 16
