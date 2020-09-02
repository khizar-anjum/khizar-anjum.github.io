---
permalink: /rcorner/projects.html
layout: page
title: Projects
collection: rcorner
---

<section>
	<div class="posts">
		{% assign sorted_posts = site.projects | sort:"date" | reverse %}
		{% for post in sorted_posts %}
			<article>
				<a href="{{post.permalink}}" class="image"><img src="{{post.poster}}" alt="" /></a>
				<h3> {{post.title}} </h3>
				<p> {{post.summary}} </p>
				<ul class="actions">
					<li><a href="{{post.permalink}}" class="button">More</a></li>
				</ul>
			</article>
		{% endfor %}
	</div>
</section>