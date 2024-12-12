== How to use variables in the documents

You can use two types of variables:
- Antora Static Variables
- JS Dynamic Variables


=== Antora Static Variables

There is a file in [content/antora.yml](content/antora.yml) where the variables used by Antora are defined. You can use them in the `adoc` documents that you create under `content/modules/ROOT` by referencing the variable name inside `{ }`, for example: 

If you have this in your `antora.yaml`

```
...
asciidoc:
  attributes:
    lab_name: "Object detection with Red Hat Device Edge"
    my_var: "foo"
...
```

And you write this in your `adoc` file:


```
This is an example:

- Lab Name: {lab_name}
- Another var: {my_var}
```

You will get this output:

```
This is an example:

- Lab Name: Object detection with Red Hat Device Edge
- Another var: foo
```

You can also use them inside code blocks in the `adoc` in this way:

```
[source,sh,role=execute,subs="attributes"]
----
cd ~/mydir/{my_var}
----
```

You will obtain this output:

```
----
cd ~/mydir/foo
----
``` 


These variables can only be changed during the documentation build time, so you cannot dynamically change them without building the guide again.


=== JS Dynamic Variables

Besides the Antora varaibles, a dynamic variable injection method was included in this guide. There is a script in [content/supplemental-ui/partials/head-scripts.hbs](content/supplemental-ui/partials/head-scripts.hbs) that get user input for some variables and then the value is reflected instantanelly in the document files.

In this case, the input of those variables is gathered in the Guide's header by using the file [content/supplemental-ui/partials/header-content.hbs](content/supplemental-ui/partials/header-content.hbs).

Imagine that we have this code in `content/supplemental-ui/partials/header-content.hbs` asking for an `username` variable:

```
...

          <div class="user-inputs">
            <label for="uname">User Name: </label><br/>
            <input type="text" id="uname" name="username" placeholder="Your username..">
          </div>
...
```

The script `content/supplemental-ui/partials/head-scripts.hbs` will get those values (also will asign a default value if it is empty):

```
  function setNamesInStorage() {
      let uname = (document.getElementById('uname').value).toLowerCase();

      if (uname === '') {
          alert('Please enter a username');
          return;
      }

      localStorage.setItem('USER_NAME', uname);
      updateMessage();
  }

    function updateMessage() {
        const username = localStorage.getItem('USER_NAME') || '{USER_NAME}';
        const unameVals = document.querySelectorAll('#unameVal');
        unameVals.forEach(element => {
            element.innerText = username;
        });       
    }
```

Then you can use those variables in the `adoc in this way`:

```
Hello user pass:[<span id="unameVal"></span>] 

I will show you a code line that you can copy: 

[source,sh,role=execute,subs="attributes"]
----
cd ~/mydir
echo "Hello <span id="unameVal"></span>"
----

and just a code without the copy option: 

[subs=quotes]
----
cd ~/mydir
echo "Hello <span id="unameVal"></span>"
----

```

Then, if you use the header input box to assign the value `myuser`, and click "Save", then the output will be:

```
Hello user myuser 

I will show you a code line that you can copy: 

----
cd ~/mydir
echo "Hello myuser"
----

and just a code without the copy option: 

----
cd ~/mydir
echo "Hello myuser"
----


```









User: pass:[<span id="unameVal"></span>]

Cluster: pass:[<span id="cdomainVal"></span>]

Git Repo: pass:[<span id="gitserverVal"></span>]



With Code:

[source,sh,role=execute,subs="attributes"]
----
cd ~/mydir
echo "Hello <span id="unameVal"></span>"
----





[subs=quotes]
----
cd ~/mydir
echo "Hello <span id="unameVal"></span>"
----




