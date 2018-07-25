# pyhks

This is a simple dependency free Python library for the Heat Kernel Signature on triangle meshes.  The only dependencies are the numpy/scipy stack.  If you want to view the results of the computation, you should also download [meshlab].

## Running
To see all options, run the script as follows
~~~~~ bash
python HKS.py --help
~~~~~

As an example, let's examine the HKS on the "homer" mesh in this repository, at different scales.  In each example, we output to a file which can be opened in [meshlab], which is the homer mesh colored in grayscale with the values of the HKS


<table>
<tr>
<td>
<code>
python HKS.py --input homer.off --t 5 --output hks5.off
</code>
</td>
<td>
<code>
python HKS.py --input homer.off --t 20 --output hks20.off
</code>
</td>
<td>
<code>
python HKS.py --input homer.off --t 200 --output hks200.off
</code>
</td>
</tr>

<tr>
<td>
<img src = "hks5.png">
</td>
<td>
<img src = "hks20.png">
</td>
<td>
<img src = "hks200.png">
</td>
</td>

</table>

Notice how at smaller time scales, finer, high frequency curvature detail is present.  However, if the time scale is too small, artifacts are present from using a limited number of eigenvectors.


[meshlab]: <http://www.meshlab.net>
