#requirements: numpy, pandas, sklearn, scipy, matplotlib, mpl_toolkits, seaborn
#todo: add assignment3data.txt.txt
import numpy as np
import pandas as pd # added pandas module for v 0.0.5 (needed when features=3, visualize=True)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import warnings #v0.0.6 added warnings

class Optimal():#v0.1.0 
    data = pd.read_txt('assignment3data.txt')
data.head()

    >>> from OptimalCluster.opticlust import Optimal
    >>> opt = Optimal({'max_iter':200})
    
    ----------
    
    """
    
    opti_df = None #check TODO
    
    def __init__(
        self,
        kmeans_kwargs: dict = None
    ):
        """
        Construct Optimal with parameter kmeans_kwargs.
        
        """
        self.kmeans_kwargs = kmeans_kwargs #check TODO
    
    def elbow(self,df,upper=15,display=False,visualize=False,function='inertia',method='angle',sq_er=1):
        """
        Determines optimal number of clusters using elbow method.
        
         Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
        
        function : {'inertia','distortion'}, default = 'inertia'
            The function used to calculate Variation
            
            'inertia' : Variation used in KMeans is inertia
            
            'distortion' : Variation used in KMeans is distortion
            
            For more information visit
            https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
            
        method : {'angle','lin'}, default = 'angle'
            The method used to calculate the optimal cluster value.
            
            'angle' : The point which has the largest angle of inflection is taken as 
                the optimal cluster value. Works well for lesser number of actual clusters.
                
            'lin' : The first point where the sum of the squares of the difference in 
                slopes of every point after it is less than the sq_er parameter is 
                taken as the optimal cluster value. Works well for a larger range of 
                actual cluster values since the sq_er parameter is adjustable. 
                For more details visit *link*
                
        sq_er : int, default = 1
            Only used when method parameter is 'lin'. The first point where the sum 
            of the squares of the difference in slopes of every point after it is 
            less than the sq_er parameter is taken as the optimal cluster value. When 
            there is suspected to be overlapping/not well defined clusters, a lower 
            value of this parameter can help separate these clusters and give a better 
            value for optimal cluster.
            For more details visit *link*
            
        ----------
        
    
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=2, centers=3)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.elbow(df)
        Optimal number of clusters is:  3  
    
        ----------
    
        """
        
        #lower is always 1, list to store inertias
        lower=1
        inertia = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        #populating inertia
        for i in K:
            
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)

            if function=='inertia':
                
                #appending KMeans inertia_ variable
                inertia.append(cls.inertia_)
            elif function=='distortion':
                
                #distortion value from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
                inertia.append(sum(
                                np.min(
                                    cdist(df, cls.cluster_centers_, 'euclidean'),axis=1
                                       )
                                    ) / df.shape[0])
            else:
                
                #for incorrect function
                raise ValueError('function should be "inertia" or "distortion"')
        
        #standardizing inertia to a fixed range
        inertia = np.array(inertia)/(np.array(inertia)).max()*14 #v0.0.6 changed to fixed number of 14 (elbow)
        
        #calulating slopes
        slopes = [inertia[0]-inertia[1]]#check TODO
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
            
        #calculating angles
        angles = []
        for i in range(len(slopes)-1):
            angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
            
        #plotting scree plot
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
            
        #finding optimal cluster value
        extra=''
        
        #using maximum angle method
        if method == 'angle':
            optimal = np.array(angles).argmax()+1
            confidence = round(np.array(angles).max()/90*100,2)#percentage of angle out of 90
            if confidence<=50:
                extra=' with Confidence:'+str(confidence)+'%.'+' Try using elbow_kf, gap_stat_se or other methods, or change the method parameter to "lin"'
        
        #using linearity method
        elif method == 'lin': #v0.0.6 changed method for lin
            flag=False
            for i in range(len(slopes)-1):
                
                #finding first point that satisfies condition
                if (sum([(slopes[i]-slopes[j])**2 for j in range(i+1,len(slopes))]))<=sq_er:
                    optimal = i
                    flag=True
                    break
            #if no point satisfies, raise warning
            if flag==False:
                optimal=upper-1
                warnings.warn("Optimal cluster value did not satisfy sq_er condition. Try increasing value of parameter upper for a better result")
        
        #for incorrect method
        else:
            raise ValueError('method should be "angle" or "lin"')
            
        #calling visualization    
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
                # raised ValueErrors for fail cases v 0.0.5
        
        #printing result
        print('Optimal number of clusters is: ',str(optimal),extra) 
        
        #return optimal cluster value
        return optimal 
    
    def elbow_kf(self,df,upper=15,display=False,visualize=False,function='inertia',se_weight=0):#v0.1.0 changed default value to 0 for se_weight param
        """
        Determines optimal number of clusters by measuring linearity 
        along with a k factor analysis.
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
        
        function : {'inertia','distortion'}, default = 'inertia'
            The function used to calculate Variation
            
            'inertia' : Variation used in KMeans is inertia
            
            'distortion' : Variation used in KMeans is distortion
            
            For more information visit
            https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
               
        se_weight : int, default = 0
            The standard error weight parameter used to find the k criteria. In general,
            this parameter is to be increased until a satisfactory k factor is reached, 
            and the corresponding value for cluster number is the optimal cluster. When 
            there are overlapping/not well defined clusters it is better to increase the 
            value of this parameter.
            Note - the value of se_weight will automatically increase by increments of 0.5
            if the k criteria is not satisfied in that iteration.
            
        ----------
